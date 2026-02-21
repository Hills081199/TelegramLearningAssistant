"""
Telegram Bot — Interface cho Learning Agents Ecosystem
Tính năng quiz: Inline keyboard buttons (A/B/C/D) + context-aware từ hội thoại
"""
import asyncio
import json
from typing import Optional
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from telegram.constants import ParseMode, ChatAction
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from config.settings import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    AGENTS_CONFIG, DAILY_REVIEW_HOUR, DAILY_QUIZ_HOUR,
)
from agents.learning_agents import get_all_agents
from agents.quiz_agent import QuizAgent
from supervisor import SupervisorAgent
from tools.learning_tools import (
    get_all_due_cards, get_learning_stats, generate_quiz
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage as LCHumanMessage
from config.settings import LLM_MODEL


class LearningBot:
    def __init__(self):
        logger.info("🚀 Initializing Learning Hub Bot...")
        self.agents = get_all_agents()
        self.supervisor = SupervisorAgent(self.agents)
        self.quiz_agent = QuizAgent()
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self.scheduler = AsyncIOScheduler(timezone="Asia/Ho_Chi_Minh")
        self._register_handlers()
        self._setup_scheduler()

        # Store pending quizzes per chat: {chat_id: {quiz_data, agent_id, topic}}
        self._pending_quizzes: dict[str, dict] = {}

        logger.info(f"✓ {len(self.agents)} agents loaded: {list(self.agents.keys())}")

    def _register_handlers(self):
        cmds = [
            ("start", self.cmd_start),
            ("help", self.cmd_help),
            ("menu", self.cmd_menu),
            ("quiz", self.cmd_quiz),
            ("random", self.cmd_random_topic),
            ("review", self.cmd_review),
            ("stats", self.cmd_stats),
            ("sync", self.cmd_sync),
            ("kb", self.cmd_kb_status),
            ("clear", self.cmd_clear_session),
        ]
        for cmd, handler in cmds:
            self.app.add_handler(CommandHandler(cmd, handler))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    def _setup_scheduler(self):
        # Nhắc ôn tập mỗi sáng
        self.scheduler.add_job(
            self.push_daily_review, 'cron',
            hour=DAILY_REVIEW_HOUR, minute=0
        )
        # Quiz buổi tối
        self.scheduler.add_job(
            self.push_evening_quiz, 'cron',
            hour=DAILY_QUIZ_HOUR, minute=0
        )

    # ── Commands ──────────────────────────────────────────────────────────────

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        agents_list = "\n".join(
            f"  {cfg.emoji} *{cfg.name}*" for cfg in AGENTS_CONFIG.values()
        )
        text = (
            "🎓 *Chào mừng đến Learning Hub\\!*\n\n"
            "Tôi có thể giúp bạn học:\n"
            f"{agents_list}\n\n"
            "💬 Chỉ cần nhắn tin bình thường — tôi sẽ tự route đến đúng chuyên gia\\!\n\n"
            "Gõ /menu để xem tất cả tính năng\\."
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = (
            "📚 *Hướng dẫn sử dụng*\n\n"
            "*Lệnh:*\n"
            "/menu — Menu chính\n"
            "/quiz \\[topic\\] — Làm quiz theo context đang chat\n"
            "/review — Ôn tập thẻ nhớ hôm nay\n"
            "/stats — Thống kê học tập\n"
            "/sync — Cập nhật tài liệu mới \\(incremental\\)\n"
            "/kb — Xem trạng thái knowledge bases\n"
            "/clear — Xóa lịch sử chat hiện tại\n\n"
            "*Cách học:*\n"
            "• Hỏi bất kỳ: _'Giải thích generator trong Python'_\n"
            "• Quiz: _'Quiz về FastAPI middleware'_\n"
            "• Lưu thẻ: _'Lưu khái niệm này vào thẻ nhớ'_\n"
            "• Ôn tập: _'Ôn tập thẻ nhớ hôm nay'_\n\n"
            f"*Lịch tự động:*\n"
            f"☀️ {DAILY_REVIEW_HOUR}:00 — Nhắc ôn tập sáng\n"
            f"🌙 {DAILY_QUIZ_HOUR}:00 — Quiz buổi tối"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)

    async def cmd_menu(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton(f"{cfg.emoji} {cfg.name}", callback_data=f"agent:{aid}")]
            for aid, cfg in AGENTS_CONFIG.items()
        ]
        keyboard += [
            [
                InlineKeyboardButton("🧠 Quiz từ hội thoại", callback_data="action:context_quiz"),
                InlineKeyboardButton("🎲 Random topic", callback_data="action:random_topic"),
            ],
            [
                InlineKeyboardButton("📖 Ôn tập hôm nay", callback_data="action:review"),
                InlineKeyboardButton("📊 Thống kê", callback_data="action:stats"),
            ],
            [
                InlineKeyboardButton("🔄 Sync tài liệu", callback_data="action:sync"),
            ],
        ]
        await update.message.reply_text(
            "🎓 *Learning Hub*\nChọn chủ đề hoặc hành động:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN,
        )

    async def cmd_quiz(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Tạo quiz từ context đang chat (hoặc topic chỉ định)"""
        chat_id = str(update.effective_chat.id)
        topic = " ".join(ctx.args) if ctx.args else None

        await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await self._send_context_quiz(update, chat_id, topic=topic)

    async def cmd_random_topic(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Gợi ý random topics từ knowledge bases"""
        chat_id = str(update.effective_chat.id)
        await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # Nếu có args, random theo KB cụ thể
        if ctx.args:
            kb_id = ctx.args[0].lower()
            if kb_id in self.agents:
                await self._send_random_topics(update, agent_filter=kb_id)
            else:
                await update.message.reply_text(
                    f"❌ Knowledge base '{kb_id}' không tồn tại.\n"
                    f"Có sẵn: {', '.join(self.agents.keys())}"
                )
        else:
            # Hiển thị menu chọn KB
            await self._send_random_menu(update)

    async def cmd_review(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._send_review(update)

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = "📊 *Thống kê học tập*\n\n"
        for aid, cfg in AGENTS_CONFIG.items():
            s = get_learning_stats(aid)
            text += (
                f"{cfg.emoji} *{cfg.name}*\n"
                f"  Quiz: {s['correct_answers']}/{s['total_quizzes']} "
                f"({s['accuracy']}% ✓) | 7 ngày: {s['week_accuracy']}%\n"
                f"  Thẻ nhớ: {s['total_review_cards']} tổng | "
                f"{s['due_today']} cần ôn hôm nay\n\n"
            )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    async def cmd_sync(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Sync tất cả KBs (chỉ embed file mới/thay đổi)"""
        msg = await update.effective_message.reply_text("🔄 Đang sync tài liệu...")
        results = []
        for aid, agent in self.agents.items():
            result = agent.sync_knowledge_base()
            cfg = AGENTS_CONFIG[aid]
            if result.get("indexed", 0) > 0 or result.get("deleted", 0) > 0:
                results.append(
                    f"{cfg.emoji} {cfg.name}: "
                    f"+{result.get('indexed',0)} mới, "
                    f"-{result.get('deleted',0)} xóa, "
                    f"={result.get('skipped',0)} bỏ qua"
                )
            else:
                results.append(f"{cfg.emoji} {cfg.name}: Không có thay đổi ✓")

        text = "✅ *Sync hoàn tất!*\n\n" + "\n".join(results)
        await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)

    async def cmd_kb_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Xem trạng thái knowledge bases"""
        text = "📂 *Knowledge Base Status*\n\n"
        for aid, agent in self.agents.items():
            cfg = AGENTS_CONFIG[aid]
            status = agent.kb_status()
            text += (
                f"{cfg.emoji} *{cfg.name}*\n"
                f"  📄 Files: {status.get('total_files', 0)} tổng | "
                f"{status.get('indexed_files', 0)} đã index\n"
                f"  🧩 Chunks: {status.get('total_chunks', 0)}\n"
            )
            if status.get("files"):
                file_list = ", ".join(status["files"][:3])
                if len(status["files"]) > 3:
                    file_list += f" ...+{len(status['files'])-3} file"
                text += f"  📁 {file_list}\n"
            text += "\n"
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    async def cmd_clear_session(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        session_id = str(update.effective_chat.id)
        self.supervisor.clear_session(session_id)
        self._pending_quizzes.pop(session_id, None)
        await update.message.reply_text("🗑️ Đã xóa lịch sử chat. Bắt đầu cuộc hội thoại mới!")

    # ── Message Handler ───────────────────────────────────────────────────────

    async def handle_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = str(update.effective_chat.id)
        text = update.message.text

        await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        try:
            response, agent_id = await self.supervisor.process(text, session_id=chat_id)
            cfg = AGENTS_CONFIG.get(agent_id)
            header = f"{cfg.emoji} *{cfg.name}*\n" if cfg else ""
            full = header + response
            await self._send_long(update, full)
        except Exception as e:
            logger.error(f"Message error: {e}", exc_info=True)
            await update.message.reply_text(f"⚠️ Có lỗi xảy ra: {str(e)[:150]}")

    # ── Callback Handler ──────────────────────────────────────────────────────

    async def handle_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data
        chat_id = str(update.effective_chat.id)

        # ── Agent info ────────────────────────────────────────────────────────
        if data.startswith("agent:"):
            aid = data.split(":")[1]
            cfg = AGENTS_CONFIG[aid]
            kb = self.agents[aid].kb_status()
            text = (
                f"{cfg.emoji} *{cfg.name}*\n\n"
                f"{cfg.description}\n\n"
                f"📚 Knowledge base: {kb.get('total_files',0)} files, "
                f"{kb.get('total_chunks',0)} chunks\n\n"
                "Hỏi tôi bất cứ điều gì về chủ đề này! 💬"
            )
            await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN)

        # ── Context quiz ──────────────────────────────────────────────────────
        elif data == "action:context_quiz":
            await query.edit_message_text("🧠 Đang tạo quiz từ hội thoại...")
            await self._send_context_quiz(update, chat_id, callback=True)

        # ── Random topic ──────────────────────────────────────────────────────
        elif data == "action:random_topic":
            # Hiển thị menu chọn KB
            await self._send_random_menu(update, callback=True)
        
        elif data.startswith("random_kb:"):
            kb_id = data.split(":", 1)[1]
            await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            if kb_id == "all":
                await self._send_random_topics(update, callback=True, agent_filter=None)
            else:
                await self._send_random_topics(update, callback=True, agent_filter=kb_id)

        # ── Ask about random topic (user tapped a topic button) ───────────────
        elif data.startswith("ask_topic:"):
            topic_text = data.split(":", 1)[1]
            # Delete hoặc edit message cũ
            try:
                await query.delete_message()
            except Exception:
                await query.edit_message_text(f"💬 Đang tìm hiểu: _{topic_text}_...")
            
            # Show typing action
            await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            
            # Process và reply
            await self._ask_about_topic(update, chat_id, topic_text)

        # ── Quiz answer (inline button A/B/C/D) ──────────────────────────────
        elif data.startswith("quiz_answer:"):
            await self._handle_quiz_answer(update, chat_id, data)

        # ── Review ────────────────────────────────────────────────────────────
        elif data == "action:review":
            await self._send_review(update, callback=True)

        elif data == "action:stats":
            await self.cmd_stats(update, ctx)

        elif data == "action:sync":
            await self.cmd_sync(update, ctx)

    # ── Quiz: Context-aware + Inline Keyboard ─────────────────────────────────

    async def _send_context_quiz(self, update: Update, chat_id: str,
                                  topic: str = None, callback: bool = False):
        """
        Tạo quiz DỰA TRÊN context đang giao tiếp, hiển thị bằng InlineKeyboard.
        """
        # 1. Lấy agent gần nhất và conversation history
        last_agent_id = self.supervisor.get_last_agent(chat_id)
        if not last_agent_id:
            # Nếu chưa chat gì → chọn agent đầu tiên
            last_agent_id = list(self.agents.keys())[0]

        agent = self.agents[last_agent_id]
        cfg = AGENTS_CONFIG[last_agent_id]

        # 2. Lấy conversation history làm context cho quiz
        conversation_history = self.supervisor.get_session_history(chat_id, last_agent_id)

        # 3. Lấy thêm RAG context nếu có topic
        rag_context = ""
        quiz_topic = topic
        if not quiz_topic:
            # Extract topic từ conversation gần nhất
            if conversation_history:
                # Lấy câu hỏi cuối cùng của user làm topic
                user_messages = [m for i, m in enumerate(conversation_history) if i % 2 == 0]
                quiz_topic = user_messages[-1][:100] if user_messages else cfg.name
            else:
                quiz_topic = cfg.name

        if agent.rag:
            docs = agent.rag.retrieve(quiz_topic, k=3)
            rag_context = "\n".join(d.page_content[:400] for d in docs)

        # Combine conversation context + RAG context
        combined_context = ""
        if conversation_history:
            combined_context += "=== HỘI THOẠI GẦN ĐÂY ===\n"
            combined_context += "\n".join(conversation_history[-6:])  # Last 3 Q&A pairs
            combined_context += "\n\n"
        if rag_context:
            combined_context += "=== TÀI LIỆU ===\n"
            combined_context += rag_context

        # 4. Tạo quiz bằng QuizAgent
        try:
            quiz_text, quiz_data = await self.quiz_agent.generate_from_context(
                agent_id=last_agent_id,
                topic=quiz_topic,
                context=combined_context,
                conversation_history=conversation_history,
            )
        except Exception as e:
            logger.error(f"Quiz generation error: {e}")
            msg_fn = self._get_reply_fn(update, callback)
            await msg_fn(f"⚠️ Không tạo được quiz: {str(e)[:100]}")
            return

        if not quiz_data or not quiz_data.get("options"):
            msg_fn = self._get_reply_fn(update, callback)
            await msg_fn("⚠️ Không tạo được quiz. Hãy chat thêm rồi thử lại!")
            return

        # 5. Lưu quiz pending
        self._pending_quizzes[chat_id] = {
            "quiz": quiz_data,
            "agent_id": last_agent_id,
            "topic": quiz_topic,
        }

        # 6. Build message + InlineKeyboard
        text = (
            f"🧠 *Quiz — {cfg.emoji} {cfg.name}*\n"
            f"📝 _Từ nội dung đang học_\n\n"
            f"*{quiz_data['question']}*\n\n"
        )
        for opt in quiz_data.get("options", []):
            text += f"{opt}\n"

        # Tạo InlineKeyboard cho A/B/C/D
        keyboard = [
            [
                InlineKeyboardButton("🅰️ A", callback_data="quiz_answer:A"),
                InlineKeyboardButton("🅱️ B", callback_data="quiz_answer:B"),
            ],
            [
                InlineKeyboardButton("©️ C", callback_data="quiz_answer:C"),
                InlineKeyboardButton("🇩 D", callback_data="quiz_answer:D"),
            ],
        ]

        msg_fn = self._get_reply_fn(update, callback)
        await msg_fn(
            text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def _handle_quiz_answer(self, update: Update, chat_id: str, data: str):
        """Xử lý khi user bấm nút A/B/C/D trên InlineKeyboard"""
        query = update.callback_query
        answer = data.split(":")[1]  # "A", "B", "C", or "D"

        pending = self._pending_quizzes.get(chat_id)
        if not pending:
            await query.edit_message_text("⏰ Quiz đã hết hạn. Dùng /quiz để tạo quiz mới!")
            return

        quiz = pending["quiz"]
        agent_id = pending["agent_id"]
        topic = pending["topic"]
        cfg = AGENTS_CONFIG.get(agent_id)

        # Evaluate
        try:
            result_text = await self.quiz_agent.evaluate_answer(
                agent_id=agent_id,
                quiz=quiz,
                user_answer=answer,
                topic=topic,
            )
        except Exception as e:
            logger.error(f"Quiz eval error: {e}")
            result_text = self._evaluate_quiz_sync(quiz, answer, agent_id, topic)

        # Build result message
        emoji_header = f"{cfg.emoji} " if cfg else ""
        full_text = (
            f"🧠 *Quiz — {emoji_header}{cfg.name if cfg else 'Quiz'}*\n\n"
            f"*{quiz['question']}*\n\n"
            f"Bạn chọn: *{answer}*\n\n"
            f"{result_text}"
        )

        # Add "Quiz tiếp" button
        keyboard = [
            [
                InlineKeyboardButton("🔄 Quiz tiếp", callback_data="action:context_quiz"),
                InlineKeyboardButton("📊 Thống kê", callback_data="action:stats"),
            ]
        ]

        # Clear pending quiz
        self._pending_quizzes.pop(chat_id, None)

        await query.edit_message_text(
            full_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    def _evaluate_quiz_sync(self, quiz: dict, answer: str, agent_id: str, topic: str) -> str:
        """Fallback đánh giá quiz nếu QuizAgent async fail"""
        from tools.learning_tools import save_quiz_result
        correct = answer.upper() == quiz.get("correct", "").upper()
        save_quiz_result(agent_id, topic, quiz["question"], answer, correct)

        if correct:
            return f"✅ *Chính xác!*\n\n💡 *Giải thích:* {quiz.get('explanation', '')}"
        else:
            return (
                f"❌ *Sai rồi!* Đáp án đúng: *{quiz['correct']}*\n\n"
                f"💡 *Giải thích:* {quiz.get('explanation', '')}"
            )

    # ── Scheduled Jobs ────────────────────────────────────────────────────────

    async def push_daily_review(self):
        if not TELEGRAM_CHAT_ID:
            return
        cards = get_all_due_cards()
        if not cards:
            msg = "☀️ *Buổi sáng tốt lành!*\n\nKhông có thẻ nhớ nào cần ôn hôm nay 🎉\nTiếp tục học tốt nhé!"
        else:
            grouped: dict[str, list] = {}
            for c in cards:
                grouped.setdefault(c["agent_id"], []).append(c)
            msg = f"☀️ *Ôn tập buổi sáng — {len(cards)} thẻ cần xem lại:*\n\n"
            for aid, agent_cards in grouped.items():
                cfg = AGENTS_CONFIG.get(aid)
                if cfg:
                    msg += f"{cfg.emoji} *{cfg.name}* ({len(agent_cards)} thẻ)\n"
                for card in agent_cards[:2]:
                    msg += f"• **{card['topic']}**: {card['content'][:150]}...\n"
                msg += "\n"
            msg += "_Dùng /review để ôn tập ngay!_"

        await self.app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN
        )

    async def push_evening_quiz(self):
        if not TELEGRAM_CHAT_ID:
            return
        import random
        aid = random.choice(list(self.agents.keys()))
        cfg = AGENTS_CONFIG[aid]
        msg = (
            f"🌙 *Quiz buổi tối — {cfg.name}* {cfg.emoji}\n\n"
            "Hãy kiểm tra kiến thức của bạn hôm nay!\n\n"
            f"_Dùng /quiz để bắt đầu._"
        )
        await self.app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN
        )

    # ── Random Topic ──────────────────────────────────────────────────────────

    async def _send_random_menu(self, update: Update, callback: bool = False):
        """
        Hiển thị menu chọn knowledge base để random topics.
        """
        keyboard = [
            [InlineKeyboardButton(
                f"{cfg.emoji} {cfg.name}",
                callback_data=f"random_kb:{aid}"
            )]
            for aid, cfg in AGENTS_CONFIG.items()
        ]
        keyboard.append([
            InlineKeyboardButton("🎲 Tất cả (Random mix)", callback_data="random_kb:all")
        ])
        
        text = (
            "🎲 *Random Topic*\n\n"
            "Chọn knowledge base để random topic:\n"
            "(hoặc random mix từ tất cả)"
        )
        
        fn = self._get_reply_fn(update, callback)
        await fn(
            text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def _send_random_topics(
        self, 
        update: Update, 
        callback: bool = False, 
        agent_filter: Optional[str] = None
    ):
        """
        Gợi ý random topics từ knowledge bases.
        Dùng LLM để extract tên topic sạch từ content chunks.
        
        Args:
            agent_filter: Nếu set, chỉ random từ agent này. None = tất cả.
        """
        all_topics = []

        # Filter agents nếu được chỉ định
        target_agents = {agent_filter: self.agents[agent_filter]} if agent_filter else self.agents

        for aid, agent in target_agents.items():
            if not agent.rag:
                continue
            cfg = AGENTS_CONFIG[aid]
            # Nếu chỉ 1 agent, lấy nhiều topic hơn
            n_topics = 5 if agent_filter else 2
            raw_topics = agent.rag.get_random_topics(n=n_topics)
            for t in raw_topics:
                t["agent_id"] = aid
                t["agent_name"] = cfg.name
                t["agent_emoji"] = cfg.emoji
            all_topics.extend(raw_topics)

        if not all_topics:
            fn = self._get_reply_fn(update, callback)
            kb_name = AGENTS_CONFIG[agent_filter].name if agent_filter else "Knowledge base"
            await fn(f"📭 {kb_name} trống. Hãy thêm tài liệu rồi /sync trước!")
            return

        # Dùng LLM để extract tên topic ngắn gọn từ content preview
        import random
        random.shuffle(all_topics)
        # Nếu filter 1 KB cụ thể → hiển thị nhiều hơn
        max_topics = 5 if agent_filter else 4
        selected = all_topics[:max_topics]

        snippets = "\n".join(
            f"{i+1}. [{t['agent_name']}] File: {t['source_file']}\nNội dung: {t['content_preview'][:150]}"
            for i, t in enumerate(selected)
        )

        try:
            llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7)
            resp = llm.invoke([LCHumanMessage(content=(
                f"Từ các đoạn nội dung dưới đây, hãy đặt 1 TÊN TOPIC NGẮN GỌN (3-8 từ) cho mỗi đoạn.\n"
                f"Trả về CHÍNH XÁC {len(selected)} dòng, mỗi dòng 1 tên topic.\n"
                f"Không đánh số, không giải thích.\n\n{snippets}"
            ))])
            topic_names = [l.strip() for l in resp.content.strip().split("\n") if l.strip()]
        except Exception as e:
            logger.warning(f"LLM topic naming failed: {e}")
            topic_names = []

        # Build message + InlineKeyboard
        if agent_filter:
            cfg = AGENTS_CONFIG[agent_filter]
            text = f"🎲 *Topics từ {cfg.emoji} {cfg.name}:*\n\n"
        else:
            text = "🎲 *Gợi ý topic để học hôm nay:*\n\n"
        keyboard = []

        for i, t in enumerate(selected):
            # Lấy tên topic từ LLM hoặc fallback từ file name
            if i < len(topic_names) and len(topic_names[i]) > 3:
                name = topic_names[i]
            else:
                # Fallback: dùng file name
                name = t["source_file"].rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()

            text += f"{t['agent_emoji']} *{name}*\n"
            text += f"  📄 _{t['source_file']}_\n\n"

            # Truncate topic cho callback_data (max 64 bytes)
            short_topic = name[:50]
            keyboard.append(
                [InlineKeyboardButton(
                    f"{t['agent_emoji']} {name}",
                    callback_data=f"ask_topic:{short_topic}",
                )]
            )

        # Button back to menu và quiz
        back_callback = f"random_kb:{agent_filter}" if agent_filter else "action:random_topic"
        keyboard.append([
            InlineKeyboardButton("🎲 Random lại", callback_data=back_callback),
            InlineKeyboardButton("🧠 Quiz ngay", callback_data="action:context_quiz"),
        ])
        if agent_filter:
            keyboard.append([
                InlineKeyboardButton("🔙 Chọn KB khác", callback_data="action:random_topic"),
            ])

        fn = self._get_reply_fn(update, callback)
        await fn(
            text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def _ask_about_topic(self, update: Update, chat_id: str, topic: str):
        """
        Khi user bấm một topic gợi ý → gửi qua supervisor để trả lời đầy đủ.
        """
        try:
            response, agent_id = await self.supervisor.process(
                f"Giải thích chi tiết về: {topic}", session_id=chat_id
            )
            cfg = AGENTS_CONFIG.get(agent_id)
            header = f"{cfg.emoji} *{cfg.name}*\n\n" if cfg else ""

            # Thêm buttons sau khi trả lời
            keyboard = [
                [InlineKeyboardButton("🧠 Quiz topic này", callback_data="action:context_quiz")],
                [InlineKeyboardButton("🎲 Random topic khác", callback_data="action:random_topic")],
            ]

            full = header + response
            if len(full) > 4000:
                full = full[:4000] + "\n\n_...xem thêm bằng cách hỏi lại._"

            # Get the right message object to reply to
            target_message = update.callback_query.message if update.callback_query else update.message
            
            try:
                await target_message.reply_text(
                    full,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                )
            except Exception:
                # Fallback without markdown if parsing fails
                await target_message.reply_text(
                    full,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                )
        except Exception as e:
            logger.error(f"Ask topic error: {e}", exc_info=True)
            target_message = update.callback_query.message if update.callback_query else update.message
            await target_message.reply_text(f"⚠️ Lỗi: {str(e)[:100]}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_reply_fn(self, update: Update, callback: bool = False):
        """Helper: trả về hàm reply phù hợp (callback edit hoặc message reply)"""
        if callback and update.callback_query:
            return update.callback_query.edit_message_text
        return update.message.reply_text

    async def _send_review(self, update: Update, callback: bool = False):
        cards = get_all_due_cards()
        if not cards:
            text = "🎉 Không có thẻ nào cần ôn hôm nay!\nHãy học thêm để tạo thẻ nhớ mới."
        else:
            text = f"📖 *Ôn tập hôm nay — {len(cards)} thẻ*\n\n"
            for card in cards[:5]:
                cfg = AGENTS_CONFIG.get(card["agent_id"])
                emoji = cfg.emoji if cfg else "📚"
                text += f"{emoji} **{card['topic']}**\n{card['content'][:250]}\n\n{'─'*25}\n\n"
            if len(cards) > 5:
                text += f"_...và {len(cards)-5} thẻ khác_"

        fn = self._get_reply_fn(update, callback)
        await fn(text, parse_mode=ParseMode.MARKDOWN)

    async def _send_long(self, update: Update, text: str, max_len: int = 4000):
        if len(text) <= max_len:
            try:
                await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(text)
            return
        for chunk in [text[i:i+max_len] for i in range(0, len(text), max_len)]:
            try:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(chunk)

    async def run_async(self):
        self.scheduler.start()
        await self.app.bot.set_my_commands([
            BotCommand("start", "Khởi động"),
            BotCommand("menu", "Menu chính"),
            BotCommand("quiz", "Làm quiz từ nội dung đang học"),
            BotCommand("random", "Gợi ý random topic để học"),
            BotCommand("review", "Ôn tập thẻ nhớ hôm nay"),
            BotCommand("stats", "Thống kê học tập"),
            BotCommand("sync", "Sync tài liệu mới (incremental)"),
            BotCommand("kb", "Xem trạng thái knowledge bases"),
            BotCommand("clear", "Xóa lịch sử chat"),
            BotCommand("help", "Hướng dẫn sử dụng"),
        ])
        logger.info("🤖 Bot đang chạy... (Ctrl+C để dừng)")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        
        # Keep the bot running
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("🛑 Bot đang dừng...")
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self.scheduler.shutdown()

    def run(self):
        asyncio.run(self.run_async())


if __name__ == "__main__":
    bot = LearningBot()
    bot.run()
