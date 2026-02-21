"""
RAG Engine with Incremental Indexing + Ensemble Retriever
==========================================================
Core features:
- Docling PDF parser (superior table/layout extraction)
- Incremental indexing: track files by fingerprint, skip unchanged
- Ensemble retriever: Vector (Chroma) + BM25 hybrid search
- File manifest for smart embedding deduplication
"""
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import EMBEDDING_MODEL, VECTOR_STORE_DIR


# ── Docling PDF Loader ────────────────────────────────────────────────────────

def _docling_pdf_loader(path: str) -> list[Document]:
    """
    Parse PDF bằng Docling — hỗ trợ tốt table, layout, OCR.
    Fallback về PyPDFLoader nếu Docling lỗi.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(path)
        md_content = result.document.export_to_markdown()

        if md_content and md_content.strip():
            return [Document(page_content=md_content, metadata={"source": path})]
        else:
            logger.warning(f"Docling returned empty for {path}, falling back to PyPDFLoader")
            return _pypdf_fallback(path)
    except ImportError:
        logger.warning("Docling not installed, falling back to PyPDFLoader")
        return _pypdf_fallback(path)
    except Exception as e:
        logger.warning(f"Docling error for {path}: {e}, falling back to PyPDFLoader")
        return _pypdf_fallback(path)


def _pypdf_fallback(path: str) -> list[Document]:
    """Fallback loader khi Docling không khả dụng"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(path).load()
    except Exception as e:
        logger.error(f"PyPDFLoader fallback also failed for {path}: {e}")
        return []


def _md_loader(path: str) -> list[Document]:
    """Load markdown as plain text"""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path})]


# ── Loader registry ──────────────────────────────────────────────────────────

LOADERS: dict[str, callable] = {
    ".pdf": _docling_pdf_loader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8").load(),
    ".md":  _md_loader,
    ".py":  lambda p: TextLoader(p, encoding="utf-8").load(),
    ".rst": lambda p: TextLoader(p, encoding="utf-8").load(),
}


# ── File fingerprinting ──────────────────────────────────────────────────────

def _file_fingerprint(path: Path) -> str:
    """
    Tạo fingerprint của file từ path + mtime + size.
    Nhanh hơn hash toàn bộ nội dung, đủ để phát hiện thay đổi.
    """
    stat = path.stat()
    raw = f"{path}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.md5(raw.encode()).hexdigest()


class FileManifest:
    """
    Track trạng thái của từng file đã được index.
    Lưu dưới dạng JSON: { "relative/path/file.pdf": "fingerprint_hash" }
    """

    def __init__(self, manifest_path: Path):
        self.path = manifest_path
        self._data: dict[str, str] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                self._data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    def is_changed(self, file_path: Path, kb_root: Path) -> bool:
        """True nếu file là mới hoặc đã thay đổi so với lần index trước"""
        key = str(file_path.relative_to(kb_root))
        current = _file_fingerprint(file_path)
        return self._data.get(key) != current

    def mark_indexed(self, file_path: Path, kb_root: Path):
        key = str(file_path.relative_to(kb_root))
        self._data[key] = _file_fingerprint(file_path)

    def remove(self, file_path: Path, kb_root: Path):
        """Xóa file khỏi manifest khi file bị xóa khỏi knowledge base"""
        key = str(file_path.relative_to(kb_root))
        self._data.pop(key, None)

    def all_tracked(self) -> set[str]:
        return set(self._data.keys())


# ── BM25 Document Store ──────────────────────────────────────────────────────

class BM25Store:
    """
    Lưu trữ raw chunks cho BM25 retriever.
    Persist dưới dạng pickle, tự động rebuild khi có thay đổi.
    """

    def __init__(self, store_path: Path):
        self.path = store_path
        self.documents: list[Document] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    self.documents = pickle.load(f)
            except Exception:
                self.documents = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.documents, f)

    def add_documents(self, docs: list[Document]):
        self.documents.extend(docs)

    def remove_by_source(self, source_key: str):
        self.documents = [d for d in self.documents if d.metadata.get("source_key") != source_key]

    def clear(self):
        self.documents = []

    def get_retriever(self, k: int = 5):
        """Tạo BM25Retriever từ documents hiện có"""
        if not self.documents:
            return None
        try:
            from langchain_community.retrievers import BM25Retriever
            retriever = BM25Retriever.from_documents(self.documents, k=k)
            return retriever
        except Exception as e:
            logger.error(f"BM25 retriever creation failed: {e}")
            return None


class RAGEngine:
    """
    RAG engine với incremental indexing + ensemble retriever.

    Tính năng:
    - Docling PDF parsing (table/layout/OCR support)
    - Incremental: chỉ embed file mới/thay đổi (FileManifest tracking)
    - Xóa chunks của file đã bị xóa khỏi KB
    - Ensemble retriever: Vector (Chroma) + BM25 hybrid search
    - Context-aware: trả về metadata đầy đủ để cite nguồn
    """

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150
    COLLECTION_PREFIX = "learning_"

    def __init__(self, agent_id: str, knowledge_base_path: str, file_extensions: list[str]):
        self.agent_id = agent_id
        self.kb_path = Path(knowledge_base_path)
        self.file_extensions = [ext.lower() for ext in file_extensions]
        self.persist_dir = Path(VECTOR_STORE_DIR) / agent_id
        self.manifest_path = self.persist_dir / "manifest.json"
        self.bm25_store_path = self.persist_dir / "bm25_store.pkl"
        self.collection_name = f"{self.COLLECTION_PREFIX}{agent_id}"

        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        )

        self.manifest = FileManifest(self.manifest_path)
        self.bm25_store = BM25Store(self.bm25_store_path)
        self.vectorstore: Optional[Chroma] = None
        self._init_vectorstore()

        # Sync: index mới, xóa cũ
        self.sync()

    # ── Khởi tạo / load vectorstore ──────────────────────────────────────────

    def _init_vectorstore(self):
        """Load hoặc tạo mới Chroma collection"""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        count = self.vectorstore._collection.count()
        logger.info(f"[{self.agent_id}] Vector store loaded — {count} chunks hiện có.")

    # ── Incremental sync ──────────────────────────────────────────────────────

    def sync(self) -> dict:
        """
        Đồng bộ knowledge base với vector store + BM25 store:
        1. Index các file mới hoặc đã thay đổi
        2. Xóa chunks của file đã bị xóa
        3. Rebuild BM25 index khi cần

        Returns: {"indexed": N, "deleted": M, "skipped": K}
        """
        current_files = self._scan_files()
        tracked_keys = self.manifest.all_tracked()
        current_keys = {str(f.relative_to(self.kb_path)) for f in current_files}

        # Files đã bị xóa khỏi KB → xóa chunks trong vectorstore + BM25
        deleted_keys = tracked_keys - current_keys
        deleted_count = 0
        for key in deleted_keys:
            self._delete_file_chunks(key)
            self.bm25_store.remove_by_source(key)
            self.manifest._data.pop(key, None)
            deleted_count += 1
            logger.info(f"[{self.agent_id}] 🗑️ Removed deleted file: {key}")

        # Files mới hoặc thay đổi → re-index
        indexed_count = 0
        skipped_count = 0
        bm25_changed = deleted_count > 0

        for file_path in current_files:
            if self.manifest.is_changed(file_path, self.kb_path):
                # Nếu file đã từng index → xóa chunks cũ trước
                key = str(file_path.relative_to(self.kb_path))
                if key in tracked_keys:
                    self._delete_file_chunks(key)
                    self.bm25_store.remove_by_source(key)
                    logger.info(f"[{self.agent_id}] 🔄 Re-indexing changed file: {key}")
                else:
                    logger.info(f"[{self.agent_id}] 📥 Indexing new file: {key}")

                success = self._index_file(file_path)
                if success:
                    self.manifest.mark_indexed(file_path, self.kb_path)
                    indexed_count += 1
                    bm25_changed = True
            else:
                skipped_count += 1
                logger.debug(f"[{self.agent_id}] ⏭️ Skipped (unchanged): {file_path.name}")

        if indexed_count > 0 or deleted_count > 0:
            self.manifest.save()
            self.bm25_store.save()

        total = self.vectorstore._collection.count()
        logger.info(
            f"[{self.agent_id}] ✅ Sync complete — "
            f"indexed: {indexed_count}, deleted: {deleted_count}, "
            f"skipped (already embedded): {skipped_count}, "
            f"total chunks: {total}"
        )
        return {"indexed": indexed_count, "deleted": deleted_count, "skipped": skipped_count}

    def _scan_files(self) -> list[Path]:
        """Tìm tất cả file hợp lệ trong knowledge base"""
        if not self.kb_path.exists():
            return []
        files = []
        for ext in self.file_extensions:
            if ext in LOADERS:
                files.extend(self.kb_path.rglob(f"*{ext}"))
        return sorted(files)

    def _index_file(self, file_path: Path) -> bool:
        """Load, chunk và embed một file vào vectorstore + BM25 store"""
        ext = file_path.suffix.lower()
        loader_fn = LOADERS.get(ext)
        if not loader_fn:
            return False
        try:
            docs = loader_fn(str(file_path))
            if not docs:
                return False

            # Gán metadata: source key để sau này tìm/xóa chunks
            rel_path = str(file_path.relative_to(self.kb_path))
            for doc in docs:
                doc.metadata["source_key"] = rel_path
                doc.metadata["agent_id"] = self.agent_id
                doc.metadata["file_name"] = file_path.name

            chunks = self.splitter.split_documents(docs)
            if not chunks:
                return False

            # Add vào cả vector store và BM25 store
            self.vectorstore.add_documents(chunks)
            self.bm25_store.add_documents(chunks)
            logger.debug(f"[{self.agent_id}]   → {len(chunks)} chunks từ {file_path.name}")
            return True

        except Exception as e:
            logger.error(f"[{self.agent_id}] Lỗi khi index {file_path.name}: {e}")
            return False

    def _delete_file_chunks(self, source_key: str):
        """Xóa tất cả chunks của một file dựa trên source_key metadata"""
        try:
            results = self.vectorstore._collection.get(
                where={"source_key": source_key}
            )
            ids = results.get("ids", [])
            if ids:
                self.vectorstore._collection.delete(ids=ids)
                logger.debug(f"[{self.agent_id}] Deleted {len(ids)} chunks for: {source_key}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Lỗi xóa chunks của {source_key}: {e}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Ensemble retrieval: Vector + BM25"""
        if self.vectorstore._collection.count() == 0:
            return []
        try:
            ensemble = self.get_ensemble_retriever(k=k)
            if ensemble:
                return ensemble.invoke(query)
            # Fallback: vector-only
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Retrieval error: {e}")
            return []

    def retrieve_with_score(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Retrieval kèm relevance score (distance thấp = liên quan cao)"""
        if self.vectorstore._collection.count() == 0:
            return []
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Retrieval error: {e}")
            return []

    def get_vector_retriever(self, k: int = 5):
        """Vector-only retriever (Chroma)"""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def get_bm25_retriever(self, k: int = 5):
        """BM25 keyword-based retriever"""
        return self.bm25_store.get_retriever(k=k)

    def get_ensemble_retriever(self, k: int = 5, vector_weight: float = 0.5):
        """
        Ensemble retriever kết hợp Vector search + BM25.
        vector_weight: trọng số cho vector search (0.0 - 1.0)
        bm25_weight = 1 - vector_weight
        """
        vector_retriever = self.get_vector_retriever(k=k)
        bm25_retriever = self.get_bm25_retriever(k=k)

        if not bm25_retriever:
            logger.debug(f"[{self.agent_id}] BM25 not available, using vector-only")
            return vector_retriever

        try:
            from langchain.retrievers import EnsembleRetriever
            ensemble = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[vector_weight, 1 - vector_weight],
            )
            logger.debug(f"[{self.agent_id}] Ensemble retriever: vector={vector_weight}, bm25={1-vector_weight}")
            return ensemble
        except ImportError:
            logger.warning("EnsembleRetriever not available, using vector-only")
            return vector_retriever

    # ── Status & management ───────────────────────────────────────────────────

    def status(self) -> dict:
        """Trả về thông tin trạng thái của knowledge base"""
        files = self._scan_files()
        total_chunks = self.vectorstore._collection.count()
        tracked = self.manifest.all_tracked()
        bm25_count = len(self.bm25_store.documents)

        # Chi tiết file nào đã embedded
        file_details = []
        for f in files:
            key = str(f.relative_to(self.kb_path))
            is_indexed = key in tracked
            file_details.append({
                "file": key,
                "indexed": is_indexed,
                "status": "✅ embedded" if is_indexed else "⏳ pending",
            })

        return {
            "agent_id": self.agent_id,
            "kb_path": str(self.kb_path),
            "total_files": len(files),
            "indexed_files": len(tracked),
            "total_chunks": total_chunks,
            "bm25_chunks": bm25_count,
            "files": [str(f.relative_to(self.kb_path)) for f in files],
            "file_details": file_details,
        }

    def force_reindex_all(self) -> dict:
        """Xóa toàn bộ và index lại từ đầu (dùng khi muốn thay đổi chunk size,...)"""
        logger.warning(f"[{self.agent_id}] Force reindex — xóa toàn bộ chunks...")
        # Xóa tất cả documents trong collection
        all_ids = self.vectorstore._collection.get()["ids"]
        if all_ids:
            self.vectorstore._collection.delete(ids=all_ids)
        # Reset manifest + BM25
        self.manifest._data = {}
        self.manifest.save()
        self.bm25_store.clear()
        self.bm25_store.save()
        # Sync lại từ đầu
        return self.sync()

    def get_random_topics(self, n: int = 3) -> list[dict]:
        """
        Lấy n topic ngẫu nhiên từ knowledge base.
        Mỗi topic gồm: snippet nội dung + file nguồn.
        Returns: [{"source_file": str, "content_preview": str}]
        """
        import random

        docs = self.bm25_store.documents
        if not docs:
            # Fallback: lấy từ Chroma
            try:
                all_data = self.vectorstore._collection.get(
                    limit=min(50, self.vectorstore._collection.count()),
                    include=["documents", "metadatas"],
                )
                if all_data["documents"]:
                    docs = [
                        Document(
                            page_content=doc,
                            metadata=meta or {},
                        )
                        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
                    ]
            except Exception:
                pass

        if not docs:
            return []

        # Sample random chunks, ưu tiên từ các file khác nhau
        by_source: dict[str, list[Document]] = {}
        for d in docs:
            src = d.metadata.get("file_name", d.metadata.get("source", "unknown"))
            by_source.setdefault(src, []).append(d)

        results = []
        sources = list(by_source.keys())
        random.shuffle(sources)

        for src in sources[:n]:
            chunk = random.choice(by_source[src])
            # Lấy 200 ký tự đầu làm preview
            preview = chunk.page_content.strip()[:200].replace("\n", " ")
            results.append({
                "source_file": src,
                "content_preview": preview,
            })

        # Nếu ít file hơn n, bổ sung thêm random chunks
        if len(results) < n and docs:
            extra = random.sample(docs, min(n - len(results), len(docs)))
            for chunk in extra:
                src = chunk.metadata.get("file_name", "unknown")
                preview = chunk.page_content.strip()[:200].replace("\n", " ")
                results.append({
                    "source_file": src,
                    "content_preview": preview,
                })

        return results[:n]
