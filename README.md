# 🎓 Personal Learning Assistant — Multi-Agent RAG System

> Hệ thống trợ lý học tập thông minh sử dụng **Multi-Agent Architecture** với **Agentic RAG**, **Ensemble Retriever**, và **Quiz Agent**.

## ✨ Tính năng chính

| Feature | Mô tả |
|---------|-------|
| 🤖 **Multi-Agent** | Supervisor route query → domain agents (Python, FastAPI, AI Agents, English) |
| 📄 **Docling PDF** | Parse PDF với table/layout/OCR extraction (fallback PyPDF) |
| 🔍 **Ensemble Retriever** | Kết hợp Vector search (Chroma) + BM25 keyword search |
| ✅ **Relevance Check** | Agent kiểm tra query có liên quan đến domain trước khi tìm kiếm |
| 🔬 **Research Agent** | Viết lại query đa góc độ, multi-retrieval thông minh |
| 🛡️ **Verification Agent** | Xác minh context đủ chính xác trước khi trả lời, auto-retry nếu không đủ |
| 🧠 **Quiz Agent** | Tạo quiz từ nội dung đang hỏi đáp, adaptive difficulty |
| 📦 **Smart Embedding** | Incremental indexing — chỉ embed file mới/thay đổi, skip unchanged |
| 📖 **Spaced Repetition** | Hệ thống thẻ nhớ SM-2 |

---

## 🏗️ Kiến trúc tổng quan

```mermaid
graph TB
    User([👤 User]) --> TG[Telegram Bot / CLI]
    TG --> SUP["🎯 Supervisor Agent<br/>(LangGraph Router)"]
    
    SUP -->|route| PA["🐍 Python Agent"]
    SUP -->|route| FA["⚡ FastAPI Agent"]
    SUP -->|route| AA["🤖 AI Agents Agent"]
    SUP -->|route| EA["🇬🇧 English Agent"]
    SUP -->|fallback| FB["📋 Fallback Menu"]
    
    PA & FA & AA & EA --> RAG["🔍 Agentic RAG Pipeline<br/>(Multi-Agent)"]
    PA & FA & AA & EA --> QA["🧠 Quiz Agent"]
    PA & FA & AA & EA --> SR["📖 Spaced Repetition"]
    
    RAG --> ER["Ensemble Retriever<br/>Vector + BM25"]
    ER --> VS[(Chroma Vector Store)]
    ER --> BM[(BM25 Index)]
    
    VS & BM --> KB[("📚 Knowledge Bases<br/>PDF / MD / TXT / PY")]
    
    style SUP fill:#4a90d9,color:#fff
    style RAG fill:#e74c3c,color:#fff
    style QA fill:#f39c12,color:#fff
    style ER fill:#27ae60,color:#fff
```

---

## 🔍 Luồng Agentic RAG Pipeline (Chi tiết)

Khi user đặt câu hỏi, pipeline sẽ chạy qua **6 agents** theo thứ tự:

```mermaid
flowchart TD
    START([🟢 User Query]) --> RC

    RC{{"1️⃣ Relevance Check Agent<br/>Kiểm tra query có liên quan?"}}
    RC -->|"✅ Relevant"| RW
    RC -->|"❌ Not Relevant"| NR["Trả lời:<br/>'Câu hỏi không liên quan<br/>đến chủ đề này'"]
    NR --> END_NR([🔴 END])
    
    RW["2️⃣ Research Agent<br/>(Query Rewriting)<br/>Viết lại thành 2-3 queries<br/>đa góc độ"]
    RW --> RET
    
    RET["3️⃣ Multi-Retrieve<br/>(Ensemble Retriever)<br/>Vector Search + BM25"]
    RET --> FIL
    
    FIL["4️⃣ Relevance Filter<br/>Lọc chunks<br/>distance < threshold"]
    FIL --> ASM
    
    ASM["5️⃣ Context Assembly<br/>Ghép context + citation"]
    ASM --> VER
    
    VER{{"6️⃣ Verification Agent<br/>Context đủ chính xác?"}}
    VER -->|"✅ Sufficient"| DONE["📤 Return verified context"]
    VER -->|"⚠️ Insufficient"| RETRY{"Retry count<br/>< MAX?"}
    
    RETRY -->|"Yes"| RW
    RETRY -->|"No (max reached)"| DONE
    
    DONE --> END([🔴 END])
    
    style RC fill:#3498db,color:#fff
    style RW fill:#9b59b6,color:#fff
    style RET fill:#27ae60,color:#fff
    style FIL fill:#f39c12,color:#fff
    style ASM fill:#e67e22,color:#fff
    style VER fill:#e74c3c,color:#fff
```

### Chi tiết từng Agent:

| # | Agent | Vai trò | Input | Output |
|---|-------|---------|-------|--------|
| 1 | **Relevance Check** | Xác minh query thuộc domain | query, agent_name | is_relevant, reason |
| 2 | **Research Agent** | Viết lại query đa góc độ | query, retry_notes | 2-3 rewritten queries |
| 3 | **Multi-Retrieve** | Ensemble search (Vector + BM25) | queries | raw_chunks (deduplicated) |
| 4 | **Relevance Filter** | Lọc chunks theo distance score | raw_chunks, query | filtered_chunks |
| 5 | **Context Assembly** | Ghép chunks + thêm citation | filtered_chunks | structured context |
| 6 | **Verification Agent** | Kiểm tra tính đầy đủ, chính xác | context, query | pass/fail + notes |

---

## 🔀 Ensemble Retriever

Kết hợp 2 phương pháp search để tăng recall:

```mermaid
flowchart LR
    Q[Query] --> VR["🧬 Vector Search<br/>(Chroma Embeddings)<br/>Semantic similarity"]
    Q --> BR["📝 BM25 Search<br/>(Keyword matching)<br/>TF-IDF based"]
    
    VR --> EN["🔀 Ensemble Retriever<br/>weights: [0.5, 0.5]"]
    BR --> EN
    
    EN --> RES["📄 Merged & Ranked<br/>Results"]
    
    style VR fill:#3498db,color:#fff
    style BR fill:#27ae60,color:#fff
    style EN fill:#9b59b6,color:#fff
```

| Retriever | Ưu điểm | Nhược điểm |
|-----------|---------|------------|
| **Vector (Chroma)** | Hiểu semantic meaning, tìm được paraphrase | Miss exact keywords |
| **BM25** | Chính xác với keyword/term cụ thể | Không hiểu semantic |
| **Ensemble** | Kết hợp cả hai → recall cao hơn | Chậm hơn chút |

---

## 🧠 Quiz Agent

Quiz Agent tạo quiz thông minh dựa trên nội dung đang hỏi đáp:

```mermaid
flowchart TD
    START([Quiz Request]) --> AC
    
    AC["Analyze Context<br/>Extract từ conversation<br/>hoặc RAG context"]
    AC -->|"evaluate mode"| EV
    AC -->|"generate mode"| DD
    
    DD["Determine Difficulty<br/>Dựa trên quiz history:<br/>- correct_rate > 80% → hard<br/>- correct_rate > 50% → medium<br/>- else → easy"]
    DD --> GQ
    
    GQ["Generate Quiz<br/>LLM tạo câu hỏi<br/>trắc nghiệm A/B/C/D"]
    GQ --> END1([📤 Quiz Output])
    
    EV["Evaluate Answer<br/>So sánh + giải thích<br/>+ lưu kết quả"]
    EV --> END2([📤 Result Output])
    
    style AC fill:#3498db,color:#fff
    style DD fill:#f39c12,color:#fff
    style GQ fill:#27ae60,color:#fff
    style EV fill:#e74c3c,color:#fff
```

---

## 📦 Smart Embedding (Incremental Indexing)

Hệ thống **không embed lại** file đã có — tiết kiệm token và thời gian:

```mermaid
flowchart TD
    SYNC["python main.py --sync"] --> SCAN["Scan knowledge_bases/"]
    SCAN --> CHECK{{"FileManifest check:<br/>fingerprint = hash(path + mtime + size)"}}
    
    CHECK -->|"File mới"| IDX["📥 Index: Docling parse → Chunk → Embed → Chroma + BM25"]
    CHECK -->|"File thay đổi"| DEL["🗑️ Delete old chunks<br/>→ Re-index"]
    CHECK -->|"File không đổi"| SKIP["⏭️ Skip<br/>(0 tokens used)"]
    CHECK -->|"File đã xóa"| REM["🗑️ Remove chunks<br/>from stores"]
    
    IDX & DEL --> SAVE["💾 Save manifest.json<br/>+ bm25_store.pkl"]
    SKIP --> DONE
    REM --> SAVE
    SAVE --> DONE([✅ Sync Complete])
    
    style CHECK fill:#f39c12,color:#fff
    style SKIP fill:#27ae60,color:#fff
    style IDX fill:#3498db,color:#fff
```

---

## 📁 Cấu trúc dự án

```
PersonalLearningAssistant/
├── main.py                    # Entry point (bot / CLI / sync / status)
├── bot.py                     # Telegram Bot interface
├── supervisor.py              # Supervisor Agent (LangGraph router)
├── requirements.txt           # Dependencies
├── .env                       # API keys & config
│
├── agents/
│   ├── base_agent.py          # Base class (RAG + Quiz + Review + Tools)
│   ├── learning_agents.py     # Domain agents (Python, FastAPI, AI, English)
│   └── quiz_agent.py          # 🆕 Dedicated Quiz Agent
│
├── rag/
│   ├── rag_engine.py          # RAG Engine (Docling + Ensemble + Manifest)
│   └── agentic_rag.py         # Multi-Agent RAG pipeline
│
├── tools/
│   └── learning_tools.py      # Quiz gen, Spaced Repetition, Stats
│
├── config/
│   └── settings.py            # Central configuration
│
├── knowledge_bases/           # 📚 Tài liệu học tập (PDF, MD, TXT, PY)
│   ├── python/
│   ├── fastapi/
│   ├── ai_agents/
│   └── english/
│
├── vector_stores/             # 💾 Chroma + BM25 + Manifest (auto-generated)
│   ├── python/
│   │   ├── manifest.json      # File tracking
│   │   └── bm25_store.pkl     # BM25 index
│   └── ...
│
└── data/
    └── learning.db            # SQLite (quiz results, flashcards)
```

---

## 🚀 Cài đặt & Sử dụng

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Cấu hình `.env`

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### 3. Thêm tài liệu vào Knowledge Base

```bash
# Copy PDF/MD/TXT vào thư mục tương ứng
cp my_python_book.pdf knowledge_bases/python/
cp fastapi_docs.md knowledge_bases/fastapi/
```

### 4. Sync & chạy

```bash
# Sync tài liệu (chỉ embed file mới/thay đổi)
python main.py --sync

# Xem trạng thái
python main.py --status

# Test qua CLI
python main.py --cli

# Chạy Telegram bot
python main.py
```

### 5. Commands

| Command | Mô tả |
|---------|--------|
| `python main.py` | Chạy Telegram bot |
| `python main.py --cli` | Test qua CLI |
| `python main.py --sync` | Sync tất cả knowledge bases |
| `python main.py --sync python` | Sync 1 agent cụ thể |
| `python main.py --status` | Xem trạng thái KB |
| `python main.py --reindex python` | Force reindex (xóa hết, embed lại) |

---

## 💬 Ví dụ sử dụng

```
You: Giải thích async/await trong Python

[python]: 
  1️⃣ Relevance Check    ✅ Relevant (Python concept)
  2️⃣ Research Agent      → 3 queries generated
  3️⃣ Ensemble Retrieve   → 8 unique chunks (Vector + BM25)
  4️⃣ Filter              → 5 chunks passed
  5️⃣ Assembly            → Context from "PYTHON PROGRAMMING NOTES.pdf"
  6️⃣ Verification        ✅ Sufficient

  Theo tài liệu "PYTHON PROGRAMMING NOTES.pdf":
  async/await là cú pháp trong Python để viết code bất đồng bộ...
  
  Muốn làm quiz về async/await không?

You: quiz

  🧠 Quiz — async/await (Độ khó: medium)
  
  **Trong Python, từ khóa `await` có thể dùng ở đâu?**
  A. Trong bất kỳ function nào
  B. Chỉ trong async function
  C. Chỉ trong class method
  D. Chỉ trong module level
  
  Reply A, B, C hoặc D

You: B

  ✅ Chính xác!
  💡 Giải thích: `await` chỉ có thể sử dụng bên trong...
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangGraph (StateGraph) |
| Vector Store | ChromaDB |
| BM25 | rank-bm25 |
| PDF Parser | Docling (fallback: PyPDF) |
| Embeddings | OpenAI text-embedding-3-small |
| Bot | python-telegram-bot |
| Database | SQLite |
