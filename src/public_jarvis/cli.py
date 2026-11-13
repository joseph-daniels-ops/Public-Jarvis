"""Interface de linha de comando (CLI) para o Public-Jarvis.

Subcomandos disponíveis:
- ask: envia um prompt para orquestração assíncrona
- remember: persiste um novo registro na memória episódica
- tasks: exibe o histórico recente da fila de tarefas
- health: diagnostica saúde e configurações sem vazar chaves de API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from public_jarvis.config import JarvisSettings, load_settings
from public_jarvis.memory import JarvisMemoryStore
from public_jarvis.orchestrator import AgentContext, JarvisOrchestrator
from public_jarvis.router import MultiLLMRouter


def _build_parser() -> argparse.ArgumentParser:
    """Cria a estrutura de argumentos da CLI com subcomandos."""
    parser = argparse.ArgumentParser(
        prog="public-jarvis",
        description="Public-Jarvis: Orquestrador de Agentes e Execução Segura",
    )
    parser.add_argument(
        "--env",
        dest="env_path",
        type=str,
        default=None,
        help="Caminho personalizado para arquivo de ambiente .env",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: ask
    ask_parser = subparsers.add_parser("ask", help="Envia um prompt para o Jarvis")
    ask_parser.add_argument("prompt", type=str, help="Prompt ou instrução a ser processada")
    ask_parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=".",
        help="Diretório de trabalho para execução de comandos (padrão: .)",
    )
    ask_parser.add_argument(
        "--user-id",
        "-u",
        type=int,
        default=None,
        help="ID do usuário para autorização",
    )

    # Subcomando: remember
    rem_parser = subparsers.add_parser(
        "remember", help="Armazena um tópico e conteúdo na memória episódica"
    )
    rem_parser.add_argument("topic", type=str, help="Tópico ou assunto da memória")
    rem_parser.add_argument("content", type=str, help="Conteúdo detalhado a memorizar")

    # Subcomando: tasks
    tasks_parser = subparsers.add_parser("tasks", help="Lista as tarefas recentes da fila")
    tasks_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Número máximo de tarefas a exibir (padrão: 20)",
    )

    # Subcomando: health
    subparsers.add_parser(
        "health",
        help="Retorna JSON estruturado sobre saúde do sistema e banco, sem expor segredos",
    )

    return parser


def handle_ask(
    args: argparse.Namespace,
    settings: JarvisSettings,
    memory: JarvisMemoryStore,
    router: MultiLLMRouter,
) -> int:
    """Executa o subcomando ask de forma assíncrona."""
    user_id = args.user_id
    if user_id is None:
        user_id = settings.allowed_users[0] if settings.allowed_users else 1

    workspace = os.path.abspath(args.workspace)
    ctx = AgentContext(user_id=user_id, workspace=workspace, is_authorized=False)
    orchestrator = JarvisOrchestrator(settings=settings, memory=memory, router=router)

    result = asyncio.run(orchestrator.handle(ctx=ctx, prompt=args.prompt))

    print(f"[{result.intent.upper()}] (Provedor: {result.provider} | Tarefa: {result.task_id[:8]})")
    for i, page in enumerate(result.pages, 1):
        if len(result.pages) > 1:
            print(f"--- Página {i}/{len(result.pages)} ---")
        print(page)

    return 0


def handle_remember(args: argparse.Namespace, memory: JarvisMemoryStore) -> int:
    """Executa o subcomando remember."""
    entry = memory.add_memory(topic=args.topic, content=args.content)
    print(f"Memória registrada com sucesso: {entry.entry_id[:8]} [Tópico: {entry.topic}]")
    return 0


def handle_tasks(args: argparse.Namespace, memory: JarvisMemoryStore) -> int:
    """Executa o subcomando tasks."""
    tasks = memory.list_tasks(limit=args.limit)
    if not tasks:
        print("Nenhuma tarefa registrada na fila.")
        return 0

    print(f"{'ID':<10} {'STATUS':<12} {'INTENÇÃO':<10} {'CRIADO EM':<22} {'PROMPT'}")
    print("-" * 80)
    for t in tasks:
        prompt_preview = t.prompt.replace("\n", " ")
        if len(prompt_preview) > 35:
            prompt_preview = prompt_preview[:32] + "..."
        print(
            f"{t.task_id[:8]:<10} {t.status:<12} {t.intent:<10} "
            f"{t.created_at[:19]:<22} {prompt_preview}"
        )
    return 0


def handle_health(settings: JarvisSettings, memory: JarvisMemoryStore) -> int:
    """Executa o subcomando health retornando JSON seguro sem expor credenciais."""
    # Contar total de memórias e tarefas
    try:
        cur = memory._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memories")
        total_memories = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM tasks")
        total_tasks = cur.fetchone()[0]
        db_connected = True
    except Exception:
        total_memories = 0
        total_tasks = 0
        db_connected = False

    health_data = {
        "status": "healthy" if db_connected else "degraded",
        "model": settings.model,
        "db_path": str(settings.db_path),
        "db_connected": db_connected,
        "has_gemini_key": bool(settings.gemini_api_key),
        "has_openrouter_key": bool(settings.openrouter_api_key),
        "has_telegram_token": bool(settings.telegram_bot_token),
        "allowed_users_count": len(settings.allowed_users),
        "total_memories": total_memories,
        "total_tasks": total_tasks,
    }

    print(json.dumps(health_data, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada principal da CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    env_path = Path(args.env_path) if args.env_path else None
    settings = load_settings(env_path=env_path)

    with JarvisMemoryStore(db_path=settings.db_path) as memory:
        router = MultiLLMRouter(
            gemini_api_key=settings.gemini_api_key,
            openrouter_api_key=settings.openrouter_api_key,
            model=settings.model,
        )

        if args.command == "ask":
            return handle_ask(args, settings, memory, router)
        elif args.command == "remember":
            return handle_remember(args, memory)
        elif args.command == "tasks":
            return handle_tasks(args, memory)
        elif args.command == "health":
            return handle_health(settings, memory)
        else:
            parser.print_help()
            return 1


if __name__ == "__main__":
    sys.exit(main())
