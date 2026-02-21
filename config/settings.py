"""
Central configuration for Learning Agents Ecosystem
"""
import os
from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).parent.parent


@dataclass
class AgentConfig:
    """Config cho mỗi learning agent"""
    name: str
    description: str
    knowledge_base_path: str
    use_rag: bool = True
    file_extensions: list = field(default_factory=lambda: [".pdf", ".txt", ".md", ".py"])
    emoji: str = "📚"
    # Agentic RAG: bật/tắt query rewriting + multi-retrieval
    use_agentic_rag: bool = True


# ════════════════════════════════════════════════════════════════
# DEFINE YOUR AGENTS HERE — Thêm agent mới chỉ cần thêm vào dict
# ════════════════════════════════════════════════════════════════
AGENTS_CONFIG: dict[str, AgentConfig] = {
    "python": AgentConfig(
        name="Python Master",
        description="Python từ cơ bản đến senior level: OOP, async, decorators, metaclasses, testing, design patterns...",
        knowledge_base_path=str(BASE_DIR / "knowledge_bases/python"),
        emoji="🐍",
        file_extensions=[".pdf", ".txt", ".md", ".py"],
    ),
    "algorithms": AgentConfig(
        name="Data Structures and Algorithms Expert",
        description="Data Structures and Algorithms: từ cơ bản đến senior level: LeetCode, system design, time/space complexity...",
        knowledge_base_path=str(BASE_DIR / "knowledge_bases/algorithms"),
        emoji="🧮",
        file_extensions=[".pdf", ".txt", ".md", ".py"],
    ),
    "fastapi": AgentConfig(
        name="FastAPI Expert",
        description="FastAPI: routing, middleware, Pydantic, dependencies, async, security, Docker deployment...",
        knowledge_base_path=str(BASE_DIR / "knowledge_bases/fastapi"),
        emoji="⚡",
    ),
    "ai_agents": AgentConfig(
        name="AI Agents Architect",
        description="AI Agents, LangChain, LangGraph, RAG pipelines, prompt engineering, tool use, evaluation...",
        knowledge_base_path=str(BASE_DIR / "knowledge_bases/ai_agents"),
        emoji="🤖",
    ),
    "english": AgentConfig(
        name="English Tutor",
        description="Tiếng Anh: grammar, vocabulary, IELTS, business English, writing skills...",
        knowledge_base_path=str(BASE_DIR / "knowledge_bases/english"),
        emoji="🇬🇧",
        file_extensions=[".pdf", ".txt", ".md"],
    ),
}

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Storage ───────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR: str = str(BASE_DIR / "vector_stores")
DATA_DIR: str = str(BASE_DIR / "data")

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Schedule ──────────────────────────────────────────────────────────────────
DAILY_REVIEW_HOUR: int = int(os.getenv("DAILY_REVIEW_HOUR", "8"))
DAILY_QUIZ_HOUR: int = int(os.getenv("DAILY_QUIZ_HOUR", "20"))
