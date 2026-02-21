"""
Agentic RAG — Multi-Agent Retrieval Pipeline
==============================================
Enhanced RAG pipeline with:
1. Relevance Check  — xác minh query có liên quan đến domain
2. Query Rewriting  — viết lại câu hỏi để tìm kiếm tốt hơn
3. Multi-retrieval  — dùng ensemble (vector + BM25) cho nhiều query
4. Relevance Filter — lọc chunks không liên quan
5. Context Assembly — ghép context có cấu trúc
6. Verification     — kiểm tra context có đủ/chính xác để trả lời

Dùng LangGraph để orchestrate toàn bộ flow.
"""
from typing import TypedDict
import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from loguru import logger

from config.settings import LLM_MODEL
from rag.rag_engine import RAGEngine


# ── State ─────────────────────────────────────────────────────────────────────

class AgenticRAGState(TypedDict):
    original_query: str
    rewritten_queries: list[str]
    raw_chunks: list[Document]
    filtered_chunks: list[Document]
    context: str                    # Context cuối cùng gửi cho LLM
    sources: list[str]              # Danh sách nguồn được cite
    agent_id: str
    # Multi-agent fields
    is_relevant: bool               # Query có liên quan đến domain?
    relevance_reason: str           # Lý do relevant/not relevant
    verification_passed: bool       # Verification agent đã verify?
    verification_notes: str         # Ghi chú từ verification
    retry_count: int                # Số lần retry retrieval


# ── Agentic RAG Graph ─────────────────────────────────────────────────────────

class AgenticRAGGraph:
    """
    LangGraph pipeline thực hiện multi-agent RAG.
    Flow: relevance_check → rewrite → retrieve → filter → assemble → verify → END
    """

    MAX_RETRIES = 1  # Số lần retry nếu verification fail

    def __init__(self, rag_engine: RAGEngine, agent_name: str):
        self.rag = rag_engine
        self.agent_name = agent_name
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.graph = self._build()

    def _build(self):
        workflow = StateGraph(AgenticRAGState)

        # Nodes
        workflow.add_node("relevance_check", self._relevance_check_node)
        workflow.add_node("rewrite_query", self._rewrite_node)
        workflow.add_node("multi_retrieve", self._retrieve_node)
        workflow.add_node("filter_chunks", self._filter_node)
        workflow.add_node("assemble_context", self._assemble_node)
        workflow.add_node("verify_answer", self._verification_node)

        # Entry
        workflow.set_entry_point("relevance_check")

        # Conditional: relevant? → rewrite : END
        workflow.add_conditional_edges(
            "relevance_check",
            self._route_after_relevance,
            {"continue": "rewrite_query", "stop": END},
        )

        # Linear flow
        workflow.add_edge("rewrite_query", "multi_retrieve")
        workflow.add_edge("multi_retrieve", "filter_chunks")
        workflow.add_edge("filter_chunks", "assemble_context")
        workflow.add_edge("assemble_context", "verify_answer")

        # Conditional: verification passed? → END : retry
        workflow.add_conditional_edges(
            "verify_answer",
            self._route_after_verification,
            {"done": END, "retry": "rewrite_query"},
        )

        return workflow.compile()

    # ── Routing functions ─────────────────────────────────────────────────────

    def _route_after_relevance(self, state: AgenticRAGState) -> str:
        return "continue" if state.get("is_relevant", True) else "stop"

    def _route_after_verification(self, state: AgenticRAGState) -> str:
        if state.get("verification_passed", True):
            return "done"
        if state.get("retry_count", 0) >= self.MAX_RETRIES:
            logger.warning(f"[{state['agent_id']}] Max retries reached, proceeding with current context")
            return "done"
        return "retry"

    # ── Node 1: Relevance Check Agent ─────────────────────────────────────────

    def _relevance_check_node(self, state: AgenticRAGState) -> dict:
        """
        Relevance Check Agent: xác minh query có liên quan đến domain.
        Nếu không liên quan → short-circuit, không cần RAG.
        """
        query = state["original_query"]

        prompt = f"""Bạn là relevance checker cho knowledge base về "{self.agent_name}".

Câu hỏi: "{query}"

Phân tích xem câu hỏi này có liên quan đến chủ đề "{self.agent_name}" không.

Trả về CHÍNH XÁC theo format (không thêm gì khác):
RELEVANT: yes/no
REASON: <giải thích ngắn gọn 1 dòng>"""

        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content.strip()

            is_relevant = "yes" in content.lower().split("relevant:")[1].split("\n")[0] if "relevant:" in content.lower() else True
            reason_match = content.split("REASON:")[-1].strip() if "REASON:" in content else ""

            logger.info(f"[{state['agent_id']}] Relevance check: {'✅ relevant' if is_relevant else '❌ not relevant'} — {reason_match[:80]}")

            return {
                "is_relevant": is_relevant,
                "relevance_reason": reason_match,
            }
        except Exception as e:
            logger.warning(f"Relevance check error: {e}, defaulting to relevant")
            return {"is_relevant": True, "relevance_reason": "Error in check, defaulting to relevant"}

    # ── Node 2: Query Rewriting (Research Agent) ──────────────────────────────

    def _rewrite_node(self, state: AgenticRAGState) -> dict:
        """
        Research Agent: viết lại query thành 2-3 câu tìm kiếm khác nhau
        để tăng khả năng tìm đúng context.
        """
        query = state["original_query"]
        retry = state.get("retry_count", 0)

        # Nếu query ngắn/đơn giản, không cần rewrite
        if len(query.split()) < 5 and retry == 0:
            return {"rewritten_queries": [query]}

        extra_instruction = ""
        if retry > 0:
            extra_instruction = f"""
CHỐNG LƯU Ý: Đây là lần retry thứ {retry}. Lần trước context chưa đủ tốt.
Verification notes: {state.get('verification_notes', '')}
Hãy viết lại queries KHÁC HOÀN TOÀN so với trước, tìm ở góc độ mới."""

        prompt = f"""Bạn là research agent chuyên tối ưu search queries cho knowledge base về {self.agent_name}.

Câu hỏi gốc: "{query}"
{extra_instruction}

Hãy viết lại thành 2-3 search queries KHÁC NHAU để tìm kiếm trong tài liệu.
Mỗi query nên khai thác góc độ khác nhau:
- Query về khái niệm chính
- Query về ví dụ/code/implementation
- Query về best practices hoặc common patterns

Chỉ trả về các queries, mỗi query một dòng, không giải thích."""

        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            lines = [l.strip() for l in resp.content.strip().split("\n") if l.strip()]
            queries = [re.sub(r"^[\d\-\.\)•*]+\s*", "", l) for l in lines if len(l) > 5]
            queries = [query] + queries[:2]  # Luôn giữ query gốc đầu tiên
            logger.debug(f"[{state['agent_id']}] Research agent queries (retry={retry}): {queries}")
            return {"rewritten_queries": queries}
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return {"rewritten_queries": [query]}

    # ── Node 3: Multi-Retrieval (Ensemble) ────────────────────────────────────

    def _retrieve_node(self, state: AgenticRAGState) -> dict:
        """
        Retrieve chunks cho từng query bằng ensemble retriever,
        deduplicate dựa trên nội dung.
        """
        seen_content: set[str] = set()
        all_chunks: list[Document] = []

        for query in state["rewritten_queries"]:
            chunks = self.rag.retrieve(query, k=4)
            for chunk in chunks:
                # Deduplicate bằng 100 ký tự đầu
                key = chunk.page_content[:100]
                if key not in seen_content:
                    seen_content.add(key)
                    all_chunks.append(chunk)

        logger.debug(
            f"[{state['agent_id']}] Ensemble retrieved {len(all_chunks)} unique chunks "
            f"from {len(state['rewritten_queries'])} queries"
        )
        return {"raw_chunks": all_chunks}

    # ── Node 4: Relevance Filter ──────────────────────────────────────────────

    def _filter_node(self, state: AgenticRAGState) -> dict:
        """
        Lọc bỏ chunks không liên quan bằng relevance scoring.
        Dùng retrieval with score để giữ lại chunks có score tốt nhất.
        """
        chunks = state["raw_chunks"]
        query = state["original_query"]

        if not chunks:
            return {"filtered_chunks": []}

        # Retrieve lại với score để rank
        scored = self.rag.retrieve_with_score(query, k=len(chunks) + 2)

        # Distance trong Chroma: nhỏ hơn = tốt hơn
        DISTANCE_THRESHOLD = 1.5
        good_chunks = [doc for doc, score in scored if score < DISTANCE_THRESHOLD]

        # Fallback: nếu lọc quá chặt thì lấy top-4
        if not good_chunks and chunks:
            good_chunks = [doc for doc, _ in scored[:4]]

        logger.debug(
            f"[{state['agent_id']}] Filtered: {len(good_chunks)}/{len(chunks)} chunks passed"
        )
        return {"filtered_chunks": good_chunks}

    # ── Node 5: Context Assembly ──────────────────────────────────────────────

    def _assemble_node(self, state: AgenticRAGState) -> dict:
        """
        Ghép chunks thành context có cấu trúc, kèm thông tin nguồn.
        """
        chunks = state["filtered_chunks"]

        if not chunks:
            return {"context": "", "sources": []}

        # Group chunks theo file source
        source_groups: dict[str, list[str]] = {}
        for chunk in chunks:
            src = chunk.metadata.get("file_name", chunk.metadata.get("source", "unknown"))
            source_groups.setdefault(src, []).append(chunk.page_content.strip())

        # Build context string
        parts = []
        sources = []
        for src, contents in source_groups.items():
            sources.append(src)
            combined = "\n\n".join(contents)
            parts.append(f"📄 **Nguồn: {src}**\n{combined}")

        context = "\n\n" + ("─" * 50) + "\n\n".join(parts)

        return {"context": context, "sources": sources}

    # ── Node 6: Verification Agent ────────────────────────────────────────────

    def _verification_node(self, state: AgenticRAGState) -> dict:
        """
        Verification Agent: kiểm tra context có đủ và chính xác để trả lời.
        Nếu không đủ → retry với queries khác.
        """
        context = state.get("context", "")
        query = state["original_query"]
        retry_count = state.get("retry_count", 0)

        # Không có context → không qua verification
        if not context:
            return {
                "verification_passed": True,
                "verification_notes": "No context available, will use general knowledge",
            }

        prompt = f"""Bạn là verification agent. Kiểm tra xem context sau có đủ thông tin
để trả lời câu hỏi hay không.

CÂU HỎI: {query}

CONTEXT:
{context[:3000]}

Đánh giá (trả về CHÍNH XÁC theo format):
SUFFICIENT: yes/no
ACCURACY: high/medium/low
NOTES: <thiếu gì, cần bổ sung gì?>"""

        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content.strip()

            sufficient = "yes" in content.lower().split("sufficient:")[1].split("\n")[0] if "sufficient:" in content.lower() else True
            notes = content.split("NOTES:")[-1].strip() if "NOTES:" in content else ""

            passed = sufficient or retry_count >= self.MAX_RETRIES

            logger.info(
                f"[{state['agent_id']}] Verification: "
                f"{'✅ passed' if passed else '⚠️ insufficient'} "
                f"(retry={retry_count}) — {notes[:80]}"
            )

            return {
                "verification_passed": passed,
                "verification_notes": notes,
                "retry_count": retry_count + 1,
            }
        except Exception as e:
            logger.warning(f"Verification error: {e}")
            return {
                "verification_passed": True,
                "verification_notes": f"Error: {e}",
                "retry_count": retry_count + 1,
            }

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, query: str, agent_id: str) -> tuple[str, list[str]]:
        """
        Chạy agentic RAG pipeline.
        Returns: (context_string, sources_list)
        """
        if self.rag.vectorstore._collection.count() == 0:
            return "", []

        initial: AgenticRAGState = {
            "original_query": query,
            "rewritten_queries": [],
            "raw_chunks": [],
            "filtered_chunks": [],
            "context": "",
            "sources": [],
            "agent_id": agent_id,
            # Multi-agent fields
            "is_relevant": True,
            "relevance_reason": "",
            "verification_passed": False,
            "verification_notes": "",
            "retry_count": 0,
        }

        result = await self.graph.ainvoke(initial)

        # Nếu not relevant, trả về context rỗng kèm lý do
        if not result.get("is_relevant", True):
            reason = result.get("relevance_reason", "")
            return f"[NOT_RELEVANT] {reason}", []

        return result["context"], result["sources"]

    def run_sync(self, query: str, agent_id: str) -> tuple[str, list[str]]:
        """Synchronous version"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.run(query, agent_id))
                    return future.result()
            else:
                return loop.run_until_complete(self.run(query, agent_id))
        except Exception as e:
            logger.error(f"AgenticRAG sync error: {e}")
            # Fallback: simple retrieve
            docs = self.rag.retrieve(query, k=4)
            context = "\n\n".join(d.page_content for d in docs)
            sources = list({d.metadata.get("file_name", "unknown") for d in docs})
            return context, sources
