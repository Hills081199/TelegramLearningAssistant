"""
Quiz Agent — Tạo quiz thông minh từ context đang hỏi đáp
=========================================================
Tạo quiz dựa trên:
- Nội dung conversation hiện tại (Q&A context)
- Tài liệu từ RAG knowledge base
- Adaptive difficulty dựa trên lịch sử quiz
"""
import json
import re
from typing import TypedDict, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from loguru import logger

from config.settings import LLM_MODEL
from tools.learning_tools import generate_quiz, save_quiz_result, get_learning_stats


# ── State ─────────────────────────────────────────────────────────────────────

class QuizState(TypedDict):
    agent_id: str
    topic: str
    context: str                    # RAG context hoặc conversation context
    conversation_history: list[str] # Lịch sử hội thoại để extract nội dung
    difficulty: str                 # easy / medium / hard
    quiz: Optional[dict]            # Quiz hiện tại
    quiz_text: str                  # Output text cho user
    mode: str                       # "generate" hoặc "evaluate"
    user_answer: str                # Câu trả lời của user (nếu evaluate)


class QuizAgent:
    """
    Dedicated Quiz Agent — tạo quiz từ context đang hỏi đáp.
    
    Modes:
    - generate: Tạo quiz mới từ topic + context
    - evaluate: Đánh giá câu trả lời của user
    - contextual: Tạo quiz từ conversation history
    """

    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.5)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(QuizState)

        workflow.add_node("analyze_context", self._analyze_context_node)
        workflow.add_node("determine_difficulty", self._difficulty_node)
        workflow.add_node("generate_quiz", self._generate_quiz_node)
        workflow.add_node("evaluate_answer", self._evaluate_node)

        workflow.set_entry_point("analyze_context")

        workflow.add_conditional_edges(
            "analyze_context",
            lambda s: "evaluate" if s.get("mode") == "evaluate" else "generate",
            {"generate": "determine_difficulty", "evaluate": "evaluate_answer"},
        )
        workflow.add_edge("determine_difficulty", "generate_quiz")
        workflow.add_edge("generate_quiz", END)
        workflow.add_edge("evaluate_answer", END)

        return workflow.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _analyze_context_node(self, state: QuizState) -> dict:
        """
        Phân tích conversation history để extract nội dung cho quiz.
        Nếu đã có context từ RAG → dùng trực tiếp.
        """
        if state.get("context") and len(state["context"]) > 50:
            return {}  # Đã có context đủ tốt

        # Extract context từ conversation history
        history = state.get("conversation_history", [])
        if not history:
            return {}

        # Lấy nội dung từ 5 messages gần nhất
        recent = history[-10:]  # 5 Q&A pairs
        combined = "\n".join(recent)

        if len(combined) > 100:
            return {"context": combined}
        return {}

    def _difficulty_node(self, state: QuizState) -> dict:
        """
        Xác định độ khó dựa trên lịch sử quiz performance.
        """
        difficulty = state.get("difficulty", "")
        if difficulty:
            return {}

        agent_id = state["agent_id"]
        try:
            stats = get_learning_stats(agent_id)
            if isinstance(stats, str):
                stats = json.loads(stats)

            total_quizzes = stats.get("total_quizzes", 0)
            correct_rate = stats.get("correct_rate", 0)

            if total_quizzes < 5:
                difficulty = "easy"
            elif correct_rate > 0.8:
                difficulty = "hard"
            elif correct_rate > 0.5:
                difficulty = "medium"
            else:
                difficulty = "easy"

            logger.debug(f"[{agent_id}] Quiz difficulty: {difficulty} (rate={correct_rate})")
        except Exception:
            difficulty = "medium"

        return {"difficulty": difficulty}

    def _generate_quiz_node(self, state: QuizState) -> dict:
        """
        Tạo quiz từ topic + context, sử dụng LLM cho chất lượng cao hơn.
        """
        topic = state.get("topic", "general")
        context = state.get("context", "")
        difficulty = state.get("difficulty", "medium")
        agent_id = state["agent_id"]

        # Dùng LLM để tạo quiz chất lượng cao từ context
        difficulty_desc = {
            "easy": "câu hỏi cơ bản, kiểm tra hiểu biết nền tảng",
            "medium": "câu hỏi trung bình, cần hiểu sâu hơn",
            "hard": "câu hỏi khó, cần phân tích và suy luận",
        }

        prompt = f"""Tạo 1 câu hỏi quiz trắc nghiệm (A/B/C/D) về chủ đề: "{topic}"
Độ khó: {difficulty} — {difficulty_desc.get(difficulty, '')}

{f'Context từ tài liệu/hội thoại:' + chr(10) + context[:2000] if context else 'Không có context cụ thể, tạo câu hỏi từ kiến thức chung.'}

Trả về JSON chính xác:
{{
    "question": "Câu hỏi...",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct": "A",
    "explanation": "Giải thích ngắn gọn tại sao đáp án đúng...",
    "difficulty": "{difficulty}"
}}"""

        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            # Parse JSON từ response
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if match:
                quiz = json.loads(match.group())
                quiz["agent_id"] = agent_id

                # Build output text
                text = f"🧠 **Quiz — {topic.title()}** (Độ khó: {difficulty})\n\n"
                text += f"**{quiz['question']}**\n\n"
                text += "\n".join(quiz.get("options", []))
                text += "\n\n_Reply A, B, C hoặc D_"

                return {"quiz": quiz, "quiz_text": text}
        except Exception as e:
            logger.error(f"Quiz generation error: {e}")

        # Fallback: dùng generate_quiz tool
        quiz = generate_quiz(topic, context, difficulty)
        if quiz.get("options"):
            text = f"🧠 **Quiz — {topic.title()}**\n\n**{quiz['question']}**\n\n"
            text += "\n".join(quiz["options"])
            text += "\n\n_Reply A, B, C hoặc D_"
        else:
            text = f"🧠 **{quiz['question']}**\n\n_Trả lời bằng lời của bạn._"

        return {"quiz": quiz, "quiz_text": text}

    def _evaluate_node(self, state: QuizState) -> dict:
        """
        Đánh giá câu trả lời của user.
        """
        quiz = state.get("quiz", {})
        answer = state.get("user_answer", "").strip().upper()[:1]
        agent_id = state["agent_id"]
        topic = state.get("topic", "")

        if not quiz or not answer:
            return {"quiz_text": "❌ Không có quiz nào đang chờ trả lời."}

        correct = answer == quiz.get("correct", "").upper()
        save_quiz_result(agent_id, topic, quiz["question"], answer, correct)

        if correct:
            text = f"✅ **Chính xác!**\n\n💡 **Giải thích:** {quiz.get('explanation', '')}"
        else:
            text = (
                f"❌ **Sai rồi!** Đáp án đúng: **{quiz['correct']}**\n\n"
                f"💡 **Giải thích:** {quiz.get('explanation', '')}\n\n"
                f"📖 Bạn muốn tôi giải thích kỹ hơn về topic này không?"
            )

        return {"quiz_text": text, "quiz": None}

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate_from_context(
        self,
        agent_id: str,
        topic: str,
        context: str = "",
        conversation_history: list[str] = None,
        difficulty: str = "",
    ) -> tuple[str, dict]:
        """
        Tạo quiz từ context hiện tại.
        Returns: (quiz_text, quiz_data)
        """
        initial: QuizState = {
            "agent_id": agent_id,
            "topic": topic,
            "context": context,
            "conversation_history": conversation_history or [],
            "difficulty": difficulty,
            "quiz": None,
            "quiz_text": "",
            "mode": "generate",
            "user_answer": "",
        }

        result = await self.graph.ainvoke(initial)
        return result.get("quiz_text", ""), result.get("quiz")

    async def evaluate_answer(
        self,
        agent_id: str,
        quiz: dict,
        user_answer: str,
        topic: str = "",
    ) -> str:
        """
        Đánh giá câu trả lời.
        Returns: response_text
        """
        initial: QuizState = {
            "agent_id": agent_id,
            "topic": topic,
            "context": "",
            "conversation_history": [],
            "difficulty": "",
            "quiz": quiz,
            "quiz_text": "",
            "mode": "evaluate",
            "user_answer": user_answer,
        }

        result = await self.graph.ainvoke(initial)
        return result.get("quiz_text", "Error evaluating answer")
