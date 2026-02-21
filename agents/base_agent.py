"""
Base Learning Agent — LangGraph-based với Agentic RAG + Quiz Agent
"""
import operator
from enum import Enum
from typing import TypedDict, Annotated, Optional, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from loguru import logger

from config.settings import AgentConfig, LLM_MODEL
from rag.rag_engine import RAGEngine
from rag.agentic_rag import AgenticRAGGraph
from agents.quiz_agent import QuizAgent
from tools.learning_tools import (
    add_review_card, get_due_cards, update_card_review,
    generate_quiz, save_quiz_result, get_learning_stats,
)


class LearningMode(str, Enum):
    CHAT = "chat"
    QUIZ = "quiz"
    REVIEW = "review"
    SUMMARY = "summary"
    EXPLAIN = "explain"


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    agent_id: str
    mode: str
    context: str
    sources: list[str]
    current_quiz: Optional[dict]
    quiz_topic: str
    metadata: dict


# Từ khóa kích hoạt quiz mode (tiếng Việt + tiếng Anh)
QUIZ_KEYWORDS = ["quiz", "kiểm tra", "test me", "đố tôi", "câu hỏi", "hỏi tôi"]
REVIEW_KEYWORDS = ["ôn tập", "review cards", "thẻ nhớ hôm nay", "spaced repetition"]
SAVE_KEYWORDS = ["lưu lại", "ghi nhớ", "thêm thẻ", "save this", "add card"]


class BaseLearningAgent:
    """
    Base class cho tất cả learning agents.
    Subclass chỉ cần override: agent_id, system_prompt.
    Tích hợp: Agentic RAG (multi-agent) + Quiz Agent + Spaced Repetition.
    """
    agent_id: str = "base"

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, streaming=False)

        # RAG engine (incremental indexing + ensemble retriever)
        self.rag = RAGEngine(
            agent_id=self.agent_id,
            knowledge_base_path=config.knowledge_base_path,
            file_extensions=config.file_extensions,
        ) if config.use_rag else None

        # Agentic RAG graph (relevance → research → verification)
        self.agentic_rag = AgenticRAGGraph(self.rag, config.name) if self.rag else None

        # Quiz Agent (dedicated quiz generation & evaluation)
        self.quiz_agent = QuizAgent()

        # Tools cho agent dùng
        self.tools = [add_review_card, get_due_cards, update_card_review]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build graph
        self.graph = self._build_graph()

    @property
    def system_prompt(self) -> str:
        return f"""Bạn là {self.config.name} {self.config.emoji} — trợ lý học tập AI chuyên sâu.
Chuyên môn: {self.config.description}

CÁCH HOẠT ĐỘNG:
- Bạn sẽ nhận được context từ tài liệu học (knowledge base) để trả lời câu hỏi
- Context đã qua pipeline: relevance check → research → verification
- Ưu tiên dùng thông tin từ context, nhưng có thể bổ sung từ kiến thức của bạn
- Khi context có thông tin, hãy cite nguồn: "Theo [tên file]..."
- Khi không có context liên quan, trả lời dựa vào kiến thức chung và nói rõ

PHONG CÁCH DẠY:
- Giải thích rõ ràng, có ví dụ thực tế
- Dùng code blocks cho code examples
- Highlight các điểm quan trọng, common mistakes
- Khi giải thích xong, gợi ý: "Muốn lưu vào thẻ nhớ không?" hoặc "Làm quiz về topic này không?"

TOOLS BẠN CÓ:
- add_review_card: lưu khái niệm vào spaced repetition khi user yêu cầu
- get_due_cards: xem thẻ cần ôn hôm nay
- update_card_review: cập nhật thẻ sau khi ôn

Trả lời cùng ngôn ngữ với user (Tiếng Việt hoặc English)."""

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("quiz", self._quiz_node)
        workflow.add_node("review", self._review_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "agent")
        workflow.add_conditional_edges(
            "agent",
            self._router,
            {"tools": "tools", "quiz": "quiz", "review": "review", "end": END},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("quiz", END)
        workflow.add_edge("review", END)

        return workflow.compile()

    # ── Nodes ──────────────────────────────────────────────────────────────────

    async def _retrieve_node(self, state: AgentState) -> AgentState:
        """Agentic RAG retrieval (multi-agent pipeline)"""
        if not self.agentic_rag:
            return {**state, "context": "", "sources": []}

        # Lấy câu hỏi cuối của user
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
        )
        if not last_human or len(last_human.strip()) < 3:
            return {**state, "context": "", "sources": []}

        # Kiểm tra intent — nếu là review/quiz không cần RAG phức tạp
        low = last_human.lower()
        if any(kw in low for kw in REVIEW_KEYWORDS):
            return {**state, "context": "", "sources": []}

        try:
            context, sources = await self.agentic_rag.run(last_human, self.agent_id)

            # Handle relevance check rejection
            if context.startswith("[NOT_RELEVANT]"):
                logger.info(f"[{self.agent_id}] Query not relevant to domain")
                return {**state, "context": context, "sources": []}

            logger.debug(f"[{self.agent_id}] RAG pipeline: {len(sources)} sources, context len={len(context)}")
            return {**state, "context": context, "sources": sources}
        except Exception as e:
            logger.error(f"[{self.agent_id}] Agentic RAG error: {e}")
            return {**state, "context": "", "sources": []}

    def _agent_node(self, state: AgentState) -> AgentState:
        """Main LLM node"""
        # Build system message với context
        context_block = ""
        context = state.get("context", "")

        if context and not context.startswith("[NOT_RELEVANT]"):
            src_list = ", ".join(state.get("sources", []))
            context_block = (
                f"\n\n━━━ KNOWLEDGE BASE CONTEXT (verified) ━━━\n"
                f"(Nguồn: {src_list})\n"
                f"{context}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )
        elif context.startswith("[NOT_RELEVANT]"):
            reason = context.replace("[NOT_RELEVANT]", "").strip()
            context_block = (
                f"\n\n⚠️ Câu hỏi không trực tiếp liên quan đến {self.config.name}."
                f"\nLý do: {reason}"
                f"\nHãy trả lời dựa trên kiến thức chung và gợi ý user hỏi đúng chủ đề.\n"
            )

        system = self.system_prompt + context_block
        messages = [SystemMessage(content=system)] + state["messages"]

        try:
            resp = self.llm_with_tools.invoke(messages)
            return {**state, "messages": [resp]}
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {**state, "messages": [AIMessage(content=f"Lỗi: {e}")]}

    async def _quiz_node(self, state: AgentState) -> AgentState:
        """Generate hoặc evaluate quiz (delegate to QuizAgent)"""
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
        )

        # Nếu đang có quiz pending và user trả lời A/B/C/D
        if state.get("current_quiz"):
            quiz = state["current_quiz"]
            answer = last_human.strip().upper()[:1]
            if answer in ("A", "B", "C", "D"):
                result_text = await self.quiz_agent.evaluate_answer(
                    agent_id=self.agent_id,
                    quiz=quiz,
                    user_answer=answer,
                    topic=state.get("quiz_topic", ""),
                )
                return {**state, "messages": [AIMessage(content=result_text)], "current_quiz": None}

        # Generate quiz mới — dùng QuizAgent
        topic = last_human
        for kw in QUIZ_KEYWORDS:
            topic = topic.lower().replace(kw, "").strip()
        topic = topic or self.config.name

        # Lấy conversation history cho context
        conv_history = [
            m.content for m in state.get("messages", [])
            if isinstance(m, (HumanMessage, AIMessage))
        ]

        context = state.get("context", "")
        if not context and self.rag:
            docs = self.rag.retrieve(topic, k=3)
            context = "\n".join(d.page_content for d in docs)

        quiz_text, quiz_data = await self.quiz_agent.generate_from_context(
            agent_id=self.agent_id,
            topic=topic,
            context=context,
            conversation_history=conv_history,
        )

        return {
            **state,
            "messages": [AIMessage(content=quiz_text)],
            "current_quiz": quiz_data,
            "quiz_topic": topic,
        }

    def _review_node(self, state: AgentState) -> AgentState:
        """Xem thẻ nhớ cần ôn hôm nay"""
        result_str = get_due_cards.invoke({"agent_id": self.agent_id})
        import json
        try:
            data = json.loads(result_str)
            cards = data.get("cards", [])
        except Exception:
            cards = []

        if not cards:
            text = f"🎉 Không có thẻ nào cần ôn tập hôm nay cho **{self.config.name}**!\n\nTiếp tục học để tạo thêm thẻ nhớ mới."
        else:
            text = f"📖 **Ôn tập hôm nay — {self.config.name}** ({len(cards)} thẻ)\n\n"
            for card in cards[:5]:
                text += f"**{card['topic']}**\n{card['content'][:300]}\n\n{'─'*30}\n\n"
            if len(cards) > 5:
                text += f"_...và {len(cards)-5} thẻ khác. Dùng /review để xem thêm._"

        return {**state, "messages": [AIMessage(content=text)]}

    # ── Router ────────────────────────────────────────────────────────────────

    def _router(self, state: AgentState) -> str:
        last_ai = state["messages"][-1]
        last_human = next(
            (m.content.lower() for m in reversed(state["messages"][:-1]) if isinstance(m, HumanMessage)),
            ""
        )

        # Tool calls
        if hasattr(last_ai, "tool_calls") and last_ai.tool_calls:
            return "tools"

        # Quiz intent
        if any(kw in last_human for kw in QUIZ_KEYWORDS):
            return "quiz"
        if state.get("current_quiz") and len(last_human.strip()) <= 2:
            return "quiz"  # User đang trả lời quiz

        # Review intent
        if any(kw in last_human for kw in REVIEW_KEYWORDS):
            return "review"

        return "end"

    # ── Public API ────────────────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        session_history: list[BaseMessage] = None,
        current_quiz: dict = None,
    ) -> tuple[str, dict]:
        """
        Chat với agent.
        Returns: (response_text, new_state_metadata)
        """
        history = list(session_history or [])
        history.append(HumanMessage(content=message))

        initial: AgentState = {
            "messages": history,
            "agent_id": self.agent_id,
            "mode": LearningMode.CHAT,
            "context": "",
            "sources": [],
            "current_quiz": current_quiz,
            "quiz_topic": "",
            "metadata": {},
        }

        result = await self.graph.ainvoke(initial)

        response = next(
            (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            "Xin lỗi, có lỗi xảy ra."
        )

        meta = {
            "current_quiz": result.get("current_quiz"),
            "sources": result.get("sources", []),
        }
        return response, meta

    def sync_knowledge_base(self) -> dict:
        """Sync KB (chỉ embed file mới/thay đổi)"""
        if not self.rag:
            return {}
        return self.rag.sync()

    def kb_status(self) -> dict:
        """Trạng thái knowledge base"""
        if not self.rag:
            return {"use_rag": False}
        return self.rag.status()

    def get_stats(self) -> dict:
        return get_learning_stats(self.agent_id)
