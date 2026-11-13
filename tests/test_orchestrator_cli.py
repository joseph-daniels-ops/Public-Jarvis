"""Testes abrangentes para o orquestrador assíncrono e CLI do Public-Jarvis."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from public_jarvis.cli import main
from public_jarvis.config import JarvisSettings
from public_jarvis.memory import JarvisMemoryStore
from public_jarvis.orchestrator import (
    AgentContext,
    JarvisOrchestrator,
    classify_intent,
    paginate_text,
)
from public_jarvis.router import LLMResponse, MultiLLMRouter


# =====================================================================
# 1. Testes de Classificação de Intenção (classify_intent)
# =====================================================================


def test_classify_intent_shell():
    assert classify_intent("!git status") == "shell"
    assert classify_intent("$ ls -la") == "shell"
    assert classify_intent("run pwd") == "shell"
    assert classify_intent("exec whoami") == "shell"
    assert classify_intent("ls -la /tmp") == "shell"
    assert classify_intent("git log -n 5") == "shell"
    assert classify_intent("df -h") == "shell"
    assert classify_intent("echo 'teste jarvis'") == "shell"
    assert classify_intent("python3 --version") == "shell"


def test_classify_intent_memory():
    assert classify_intent("Lembrar que o servidor de banco está em 10.0.0.1") == "memory"
    assert classify_intent("Favor guardar a configuração de porta 8080") == "memory"
    assert classify_intent("Memorizar a chave de ativação para depois") == "memory"
    assert classify_intent("Lembre-se de verificar o backup amanhã") == "memory"


def test_classify_intent_code():
    assert classify_intent("Escreva uma função python para ordenar listas") == "code"
    assert classify_intent("Refatore este script para melhorar tipagem") == "code"
    assert classify_intent("Como corrigir este bug de sintaxe?") == "code"
    assert classify_intent("Crie um teste unitário com pytest") == "code"


def test_classify_intent_analysis():
    assert classify_intent("Analise os logs de erro da aplicação") == "analysis"
    assert classify_intent("Diagnóstico de performance e uso de CPU") == "analysis"
    assert classify_intent("Auditar o consumo de memória RAM") == "analysis"
    assert classify_intent("Gerar um relatório de benchmark do sistema") == "analysis"


def test_classify_intent_general():
    assert classify_intent("Olá Jarvis, tudo bem?") == "general"
    assert classify_intent("Qual é a capital da França?") == "general"
    assert classify_intent("") == "general"
    assert classify_intent("   ") == "general"


# =====================================================================
# 2. Testes de Paginação de Texto (paginate_text)
# =====================================================================


def test_paginate_text_empty():
    assert paginate_text("") == [""]
    assert paginate_text("   ") == ["   "]


def test_paginate_text_short():
    short = "Este é um texto curto dentro do limite."
    pages = paginate_text(short, max_chars=100)
    assert len(pages) == 1
    assert pages[0] == short


def test_paginate_text_long_split():
    long_text = "\n\n".join([f"Parágrafo {i}: " + ("x" * 200) for i in range(10)])
    pages = paginate_text(long_text, max_chars=500)
    assert len(pages) > 1
    for page in pages:
        assert len(page) <= 500


def test_paginate_text_invalid_max_chars():
    with pytest.raises(ValueError):
        paginate_text("texto", max_chars=0)
    with pytest.raises(ValueError):
        paginate_text("texto", max_chars=-10)


# =====================================================================
# 3. Testes do Orquestrador (JarvisOrchestrator)
# =====================================================================


def test_orchestrator_safe_shell_execution(tmp_path: Path):
    memory = JarvisMemoryStore(tmp_path / "test.db")
    settings = JarvisSettings()
    router = MultiLLMRouter()
    orchestrator = JarvisOrchestrator(settings=settings, memory=memory, router=router)

    ctx = AgentContext(user_id=1, workspace=str(tmp_path), is_authorized=True)
    result = asyncio.run(orchestrator.handle(ctx=ctx, prompt='echo "hello jarvis"'))

    assert result.intent == "shell"
    assert result.provider == "shell"
    assert "hello jarvis" in result.response
    assert len(result.pages) >= 1

    tasks = memory.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].status == "completed"
    assert "hello jarvis" in tasks[0].result


def test_orchestrator_dangerous_shell_blocking(tmp_path: Path):
    memory = JarvisMemoryStore(tmp_path / "test.db")
    settings = JarvisSettings()
    router = MultiLLMRouter()
    orchestrator = JarvisOrchestrator(settings=settings, memory=memory, router=router)

    ctx = AgentContext(user_id=1, workspace=str(tmp_path), is_authorized=True)

    # Tentativa de injeção e comando proibido
    result = asyncio.run(orchestrator.handle(ctx=ctx, prompt="rm -rf /"))
    assert result.intent == "shell"
    assert result.provider == "safe_shell"
    assert "Bloqueio de segurança" in result.response

    tasks = memory.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].status == "failed"


def test_orchestrator_memory_search_and_router_fallback(tmp_path: Path):
    memory = JarvisMemoryStore(tmp_path / "test.db")
    memory.add_memory("servidor_prod", "O servidor de produção roda na porta 9090 em 10.0.0.5")

    settings = JarvisSettings()
    router = MultiLLMRouter()
    orchestrator = JarvisOrchestrator(settings=settings, memory=memory, router=router)

    ctx = AgentContext(user_id=1, workspace=str(tmp_path), is_authorized=True)
    result = asyncio.run(orchestrator.handle(ctx=ctx, prompt="Qual a porta do servidor de produção?"))

    assert result.provider == "local_fallback"
    assert "10.0.0.5" in result.response
    assert "9090" in result.response

    tasks = memory.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].status == "completed"


def test_orchestrator_user_authorization_rejected(tmp_path: Path):
    memory = JarvisMemoryStore(tmp_path / "test.db")
    settings = JarvisSettings(allowed_users=[100, 200])
    router = MultiLLMRouter()
    orchestrator = JarvisOrchestrator(settings=settings, memory=memory, router=router)

    # Usuário não autorizado 999
    ctx = AgentContext(user_id=999, workspace=str(tmp_path), is_authorized=False)
    result = asyncio.run(orchestrator.handle(ctx=ctx, prompt="echo 'ataque'"))

    assert "Acesso negado" in result.response
    assert result.provider == "system"

    tasks = memory.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].status == "rejected"


def test_orchestrator_user_authorization_accepted(tmp_path: Path):
    memory = JarvisMemoryStore(tmp_path / "test.db")
    settings = JarvisSettings(allowed_users=[100, 200])
    router = MultiLLMRouter()
    orchestrator = JarvisOrchestrator(settings=settings, memory=memory, router=router)

    # Usuário autorizado 100
    ctx = AgentContext(user_id=100, workspace=str(tmp_path), is_authorized=False)
    result = asyncio.run(orchestrator.handle(ctx=ctx, prompt="echo 'autorizado'"))

    assert "autorizado" in result.response
    assert result.provider == "shell"
    assert ctx.is_authorized is True


# =====================================================================
# 4. Testes da CLI (public_jarvis.cli)
# =====================================================================


def test_cli_health(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    # Isolar variáveis de ambiente
    for key in ["OPENROUTER_API_KEY", "TELEGRAM_BOT_TOKEN"]:
        monkeypatch.delenv(key, raising=False)

    db_file = tmp_path / "jarvis_health.sqlite"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"JARVIS_DB_PATH={db_file}\n"
        "GEMINI_API_KEY=AIzaSyFakeSecretKey123\n"
        "ALLOWED_USERS=10,20,30\n"
    )

    exit_code = main(["--env", str(env_file), "health"])
    assert exit_code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert data["status"] == "healthy"
    assert data["has_gemini_key"] is True
    assert data["has_openrouter_key"] is False
    assert data["allowed_users_count"] == 3
    # Nenhuma chave secreta revelada no JSON de saúde!
    assert "AIzaSyFakeSecretKey123" not in captured.out


def test_cli_remember_and_tasks(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    db_file = tmp_path / "jarvis_cli.sqlite"
    env_file = tmp_path / ".env"
    env_file.write_text(f"JARVIS_DB_PATH={db_file}\n")

    # 1. Subcomando remember
    ret = main(["--env", str(env_file), "remember", "infra", "Docker host em 192.168.1.10"])
    assert ret == 0
    cap1 = capsys.readouterr()
    assert "Memória registrada com sucesso" in cap1.out
    assert "infra" in cap1.out

    # 2. Subcomando tasks
    ret2 = main(["--env", str(env_file), "tasks"])
    assert ret2 == 0
    cap2 = capsys.readouterr()
    # Nenhuma tarefa adicionada ainda
    assert "Nenhuma tarefa registrada" in cap2.out


def test_cli_ask(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    db_file = tmp_path / "jarvis_ask.sqlite"
    env_file = tmp_path / ".env"
    env_file.write_text(f"JARVIS_DB_PATH={db_file}\n")

    # Ask executa shell echo seguro
    ret = main(["--env", str(env_file), "ask", 'echo "execucao cli"'])
    assert ret == 0
    cap = capsys.readouterr()
    assert "[SHELL]" in cap.out
    assert "execucao cli" in cap.out

    # Checar se task foi registrada
    ret_tasks = main(["--env", str(env_file), "tasks"])
    assert ret_tasks == 0
    cap_tasks = capsys.readouterr()
    assert "completed" in cap_tasks.out
    assert "echo" in cap_tasks.out
