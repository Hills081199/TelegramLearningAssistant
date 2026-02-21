"""
Learning Tools
==============
- Spaced Repetition (SM-2 algorithm)
- Quiz Generator
- Learning Statistics
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from loguru import logger

from config.settings import LLM_MODEL, DATA_DIR

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = str(Path(DATA_DIR) / "learning.db")


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS review_cards (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id      TEXT    NOT NULL,
            topic         TEXT    NOT NULL,
            content       TEXT    NOT NULL,
            next_review   TEXT    NOT NULL,
            interval_days INTEGER DEFAULT 1,
            ease_factor   REAL    DEFAULT 2.5,
            created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS quiz_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id    TEXT    NOT NULL,
            topic       TEXT,
            question    TEXT    NOT NULL,
            user_answer TEXT,
            correct     INTEGER DEFAULT 0,
            difficulty  TEXT    DEFAULT 'medium',
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


init_db()


# ── Spaced Repetition Tools (dùng bởi agent thông qua tool_calls) ─────────────

@tool
def add_review_card(agent_id: str, topic: str, content: str) -> str:
    """
    Lưu một khái niệm quan trọng vào hệ thống spaced repetition.
    Dùng khi user học được điều gì đó đáng nhớ.
    """
    conn = sqlite3.connect(DB_PATH)
    next_review = (datetime.now() + timedelta(days=1)).isoformat()
    conn.execute(
        "INSERT INTO review_cards (agent_id, topic, content, next_review) VALUES (?,?,?,?)",
        (agent_id, topic, content, next_review)
    )
    conn.commit()
    conn.close()
    return f"✅ Đã lưu thẻ nhớ: **{topic}** — sẽ ôn tập vào ngày mai!"


@tool
def get_due_cards(agent_id: str) -> str:
    """
    Lấy danh sách thẻ nhớ cần ôn tập hôm nay cho agent này.
    Trả về JSON list hoặc thông báo nếu không có thẻ nào.
    """
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    rows = conn.execute(
        "SELECT id, topic, content FROM review_cards WHERE agent_id=? AND next_review<=? ORDER BY next_review",
        (agent_id, now)
    ).fetchall()
    conn.close()

    if not rows:
        return json.dumps({"cards": [], "message": "Không có thẻ nào cần ôn tập hôm nay! 🎉"})
    cards = [{"id": r[0], "topic": r[1], "content": r[2]} for r in rows]
    return json.dumps({"cards": cards, "message": f"Có {len(cards)} thẻ cần ôn hôm nay."})


@tool
def update_card_review(card_id: int, quality: int) -> str:
    """
    Cập nhật thẻ nhớ sau khi ôn tập (SM-2 algorithm).
    quality: 0-5 (0 = quên hoàn toàn, 3 = nhớ vất vả, 5 = nhớ hoàn hảo)
    """
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT interval_days, ease_factor FROM review_cards WHERE id=?", (card_id,)
    ).fetchone()

    if not row:
        conn.close()
        return "❌ Không tìm thấy thẻ này."

    interval, ef = row

    # SM-2 Algorithm
    if quality < 3:
        interval = 1  # Reset về ngày mai nếu quên
    elif interval <= 1:
        interval = 3
    elif interval <= 3:
        interval = 7
    else:
        interval = round(interval * ef)

    # Cập nhật ease factor
    ef = max(1.3, ef + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    next_review = (datetime.now() + timedelta(days=interval)).isoformat()

    conn.execute(
        "UPDATE review_cards SET interval_days=?, ease_factor=?, next_review=? WHERE id=?",
        (interval, ef, next_review, card_id)
    )
    conn.commit()
    conn.close()
    return f"📅 Đã cập nhật! Ôn lại sau **{interval} ngày**."


# ── Quiz ──────────────────────────────────────────────────────────────────────

def generate_quiz(topic: str, context: str, difficulty: str = "medium") -> dict:
    """Tạo câu hỏi quiz từ context tài liệu"""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.8)

    diff_guide = {
        "easy": "câu hỏi định nghĩa cơ bản, nhận dạng khái niệm",
        "medium": "câu hỏi áp dụng, so sánh, giải thích cơ chế",
        "hard": "câu hỏi phân tích sâu, edge case, trade-offs, best practices",
    }.get(difficulty, "câu hỏi áp dụng")

    prompt = f"""Bạn là giáo viên tạo câu hỏi kiểm tra kiến thức về: {topic}

Độ khó: {difficulty} — {diff_guide}

Nội dung tài liệu tham khảo:
{context[:2000]}

Hãy tạo 1 câu hỏi trắc nghiệm. Trả về JSON:
{{
  "question": "câu hỏi rõ ràng, cụ thể",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct": "A",
  "explanation": "giải thích tại sao đáp án đúng, và tại sao các đáp án khác sai"
}}

Chỉ trả về JSON, không thêm text nào khác."""

    try:
        resp = llm.invoke(prompt)
        # Clean markdown code block nếu có
        content = resp.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        return {
            "question": f"Giải thích khái niệm: {topic}",
            "options": [],
            "correct": "",
            "explanation": "Hãy trả lời bằng lời của bạn."
        }


def save_quiz_result(agent_id: str, topic: str, question: str,
                     user_answer: str, correct: bool, difficulty: str = "medium"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO quiz_results (agent_id, topic, question, user_answer, correct, difficulty) VALUES (?,?,?,?,?,?)",
        (agent_id, topic, question, user_answer, int(correct), difficulty)
    )
    conn.commit()
    conn.close()


def get_learning_stats(agent_id: str) -> dict:
    """Thống kê học tập của một agent"""
    conn = sqlite3.connect(DB_PATH)

    total, correct_sum = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(correct),0) FROM quiz_results WHERE agent_id=?", (agent_id,)
    ).fetchone()

    total_cards, = conn.execute(
        "SELECT COUNT(*) FROM review_cards WHERE agent_id=?", (agent_id,)
    ).fetchone()

    due_today, = conn.execute(
        "SELECT COUNT(*) FROM review_cards WHERE agent_id=? AND next_review<=?",
        (agent_id, datetime.now().isoformat())
    ).fetchone()

    # Thống kê 7 ngày gần nhất
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    week_total, week_correct = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(correct),0) FROM quiz_results WHERE agent_id=? AND created_at>=?",
        (agent_id, week_ago)
    ).fetchone()

    conn.close()
    return {
        "total_quizzes": total,
        "correct_answers": correct_sum,
        "accuracy": round(correct_sum / total * 100, 1) if total > 0 else 0,
        "total_review_cards": total_cards,
        "due_today": due_today,
        "week_quizzes": week_total,
        "week_accuracy": round(week_correct / week_total * 100, 1) if week_total > 0 else 0,
    }


def get_all_due_cards() -> list[dict]:
    """Lấy tất cả thẻ cần ôn hôm nay của mọi agent"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, agent_id, topic, content FROM review_cards WHERE next_review<=? ORDER BY agent_id",
        (datetime.now().isoformat(),)
    ).fetchall()
    conn.close()
    return [{"id": r[0], "agent_id": r[1], "topic": r[2], "content": r[3]} for r in rows]
