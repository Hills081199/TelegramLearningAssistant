"""
Learning Agents Ecosystem — Entry Point

Commands:
    python main.py              → Chạy Telegram bot
    python main.py --cli        → Test qua CLI (không cần Telegram)
    python main.py --sync       → Sync tất cả knowledge bases
    python main.py --status     → Xem trạng thái KB
    python main.py --reindex <agent_id>  → Force reindex một agent
"""
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def run_bot():
    from bot import LearningBot
    bot = LearningBot()
    bot.run()


def run_sync(agent_ids: list[str] = None):
    """Sync KBs — chỉ embed file mới/thay đổi"""
    from learning_agents import get_all_agents
    from settings import AGENTS_CONFIG

    agents = get_all_agents()
    targets = agent_ids or list(agents.keys())

    print("\n📚 Syncing Knowledge Bases (incremental)\n" + "="*45)
    total_indexed = 0
    for aid in targets:
        if aid not in agents:
            print(f"⚠ Unknown agent: {aid}")
            continue
        cfg = AGENTS_CONFIG[aid]
        print(f"\n{cfg.emoji} {cfg.name}...")
        result = agents[aid].sync_knowledge_base()
        print(f"  ✓ Indexed: {result.get('indexed',0)} new/changed files")
        print(f"  ✓ Deleted: {result.get('deleted',0)} removed files")
        print(f"  ✓ Skipped: {result.get('skipped',0)} unchanged files")
        total_indexed += result.get('indexed', 0)
    print(f"\n✅ Done! {total_indexed} files newly indexed.\n")


def run_status():
    """Xem trạng thái tất cả KBs"""
    from learning_agents import get_all_agents
    from settings import AGENTS_CONFIG

    agents = get_all_agents()
    print("\n📂 Knowledge Base Status\n" + "="*45)
    for aid, agent in agents.items():
        cfg = AGENTS_CONFIG[aid]
        status = agent.kb_status()
        print(f"\n{cfg.emoji} {cfg.name} ({aid})")
        print(f"  Path: {status.get('kb_path', 'N/A')}")
        print(f"  Files: {status.get('total_files', 0)} total | "
              f"{status.get('indexed_files', 0)} indexed")
        print(f"  Chunks: {status.get('total_chunks', 0)}")
        if status.get("files"):
            for f in status["files"]:
                print(f"    - {f}")
    print()


def run_reindex(agent_id: str):
    """Force reindex toàn bộ một agent (dùng khi thay đổi chunk size)"""
    from learning_agents import create_agent
    from settings import AGENTS_CONFIG

    if agent_id not in AGENTS_CONFIG:
        print(f"❌ Unknown agent: {agent_id}")
        return

    cfg = AGENTS_CONFIG[agent_id]
    print(f"\n{cfg.emoji} Force reindexing {cfg.name}...")
    agent = create_agent(agent_id)
    result = agent.rag.force_reindex_all()
    print(f"✅ Done: {result.get('indexed', 0)} files indexed, "
          f"{result.get('skipped', 0)} skipped\n")


async def run_cli():
    """Interactive CLI cho test"""
    from learning_agents import get_all_agents
    from supervisor import SupervisorAgent

    print("\n🎓 Learning Hub — CLI Mode")
    print("Agents:", list(get_all_agents().keys()))
    print("Type 'quit' to exit, 'sync' to sync KBs, 'status' for KB status\n")

    agents = get_all_agents()
    supervisor = SupervisorAgent(agents)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "sync":
                run_sync()
                continue
            if user_input.lower() == "status":
                run_status()
                continue

            response, agent_id = await supervisor.process(user_input, session_id="cli")
            print(f"\n[{agent_id}]: {response}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--cli" in args:
        asyncio.run(run_cli())
    elif "--sync" in args:
        # Optional: python main.py --sync python fastapi
        targets = [a for a in args if not a.startswith("--")]
        run_sync(targets if targets else None)
    elif "--status" in args:
        run_status()
    elif "--reindex" in args:
        idx = args.index("--reindex")
        if idx + 1 < len(args):
            run_reindex(args[idx + 1])
        else:
            print("Usage: python main.py --reindex <agent_id>")
    else:
        run_bot()
