"""
Supervisor Agent — Orchestrate toàn bộ learning agents
"""
import json
import operator
import re
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from loguru import logger

from config.settings import LLM_MODEL, AGENTS_CONFIG
from agents.base_agent import BaseLearningAgent


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    selected_agent: str
    response: str
    session_id: str
    # Quiz state per-session per-agent
    quiz_states: dict    # {agent_id: current_quiz}


ROUTE_PROMPT = """Bạn là học tập hub supervisor. Phân tích tin nhắn và route đến agent phù hợp.

Agents có sẵn:
{agents_list}

Trả về JSON:
{{
  "agent_id": "<một trong các agent IDs trên>",
  "confidence": 0.0-1.0,
  "reasoning": "lý do chọn agent này"
}}

Nếu tin nhắn không rõ ràng về chủ đề → chọn agent có confidence thấp nhất và để fallback xử lý."""


class SupervisorAgent:
    def __init__(self, agents: dict[str, BaseLearningAgent]):
        self.agents = agents
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.graph = self._build_graph()

        # Session state: {session_id: {agent_id: {history, current_quiz}}}
        self._sessions: dict[str, dict] = {}

    def _build_graph(self):
        workflow = StateGraph(SupervisorState)
        workflow.add_node("route", self._route_node)
        workflow.add_node("delegate", self._delegate_node)
        workflow.add_node("fallback", self._fallback_node)

        workflow.set_entry_point("route")
        workflow.add_conditional_edges(
            "route",
            lambda s: "delegate" if s.get("selected_agent") else "fallback",
            {"delegate": "delegate", "fallback": "fallback"},
        )
        workflow.add_edge("delegate", END)
        workflow.add_edge("fallback", END)
        return workflow.compile()

    def _route_node(self, state: SupervisorState) -> SupervisorState:
        agents_list = "\n".join(
            f'  - "{aid}": {cfg.name} — {cfg.description[:100]}'
            for aid, cfg in AGENTS_CONFIG.items()
        )
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
        )

        prompt = ROUTE_PROMPT.format(agents_list=agents_list)
        try:
            resp = self.llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=f'User: "{last_human}"')
            ])
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                aid = data.get("agent_id", "")
                confidence = data.get("confidence", 0)
                if aid in self.agents and confidence >= 0.5:
                    logger.info(f"Routing → {aid} (confidence={confidence:.2f})")
                    return {**state, "selected_agent": aid}
        except Exception as e:
            logger.error(f"Routing error: {e}")

        return {**state, "selected_agent": ""}

    async def _delegate_node(self, state: SupervisorState) -> SupervisorState:
        agent_id = state["selected_agent"]
        agent = self.agents[agent_id]
        session_id = state["session_id"]

        # Lấy session state cho agent này
        session = self._sessions.setdefault(session_id, {})
        agent_session = session.setdefault(agent_id, {"history": [], "current_quiz": None})

        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
        )

        response, meta = await agent.chat(
            message=last_human,
            session_history=agent_session["history"],
            current_quiz=agent_session.get("current_quiz"),
        )

        # Update session
        agent_session["history"].append(HumanMessage(content=last_human))
        agent_session["history"].append(AIMessage(content=response))
        agent_session["history"] = agent_session["history"][-20:]  # Keep last 20
        agent_session["current_quiz"] = meta.get("current_quiz")

        return {**state, "response": response}

    def _fallback_node(self, state: SupervisorState) -> SupervisorState:
        menu = "\n".join(
            f"  {cfg.emoji} *{cfg.name}* — /{aid}"
            for aid, cfg in AGENTS_CONFIG.items()
        )
        response = (
            "👋 Chào! Tôi là Learning Hub của bạn.\n\n"
            f"📚 Chủ đề học tập:\n{menu}\n\n"
            "💡 *Cách dùng:*\n"
            "• Hỏi thẳng: _'Giải thích async/await trong Python'_\n"
            "• Quiz: _'Quiz về FastAPI dependencies'_\n"
            "• Ôn tập: _'Ôn tập thẻ nhớ hôm nay'_\n"
            "• Lưu: _'Lưu khái niệm này vào thẻ nhớ'_"
        )
        return {**state, "response": response}

    async def process(self, message: str, session_id: str = "default") -> tuple[str, str]:
        """Returns (response, agent_id_used)"""
        initial: SupervisorState = {
            "messages": [HumanMessage(content=message)],
            "selected_agent": "",
            "response": "",
            "session_id": session_id,
            "quiz_states": {},
        }
        result = await self.graph.ainvoke(initial)
        return result["response"], result.get("selected_agent", "supervisor")

    def get_session_history(self, session_id: str, agent_id: str = None) -> list[str]:
        """
        Lấy conversation history text cho quiz context.
        Nếu agent_id=None → lấy tất cả agents trong session.
        Returns: list of message content strings.
        """
        session = self._sessions.get(session_id, {})
        if not session:
            return []

        texts = []
        agents_to_check = [agent_id] if agent_id else list(session.keys())
        for aid in agents_to_check:
            agent_data = session.get(aid, {})
            for msg in agent_data.get("history", [])[-10:]:
                texts.append(msg.content)
        return texts

    def get_last_agent(self, session_id: str) -> str:
        """Trả về agent_id được dùng gần nhất trong session"""
        session = self._sessions.get(session_id, {})
        if not session:
            return ""
        # Return last agent that has history
        for aid in reversed(list(session.keys())):
            if session[aid].get("history"):
                return aid
        return ""

    def clear_session(self, session_id: str):
        """Xóa toàn bộ lịch sử chat của session"""
        self._sessions.pop(session_id, None)
