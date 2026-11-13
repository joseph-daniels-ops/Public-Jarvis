"""Testes unitários e de integração para os módulos memory e router do Public-Jarvis."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from public_jarvis.memory import (
    JarvisMemoryStore,
    MemoryEntry,
    TaskItem,
    cosine_similarity,
    deterministic_vectorize,
)
from public_jarvis.router import LLMResponse, MultiLLMRouter


# =====================================================================
# 1. Testes de Vetorização Determinística e Similaridade de Cosseno
# =====================================================================


def test_deterministic_vectorize_consistency() -> None:
    """Verifica se o vetor gerado é 100% determinístico e consistente entre execuções."""
    text = "Jarvis autonomous agent orchestrator with multi-LLM routing"
    v1 = deterministic_vectorize(text, dim=128)
    v2 = deterministic_vectorize(text, dim=128)
    assert v1 == v2
    assert len(v1) == 128


def test_deterministic_vectorize_l2_normalization() -> None:
    """Garante que a norma L2 do vetor resultante é aproximadamente 1.0."""
    v = deterministic_vectorize("Engenharia de software com TDD e SQLite", dim=128)
    norm_sq = sum(x * x for x in v)
    assert math.isclose(norm_sq, 1.0, rel_tol=1e-5)


def test_deterministic_vectorize_empty_and_spaces() -> None:
    """Verifica que strings vazias ou apenas espaços retornam vetor nulo."""
    v_empty = deterministic_vectorize("", dim=64)
    assert v_empty == [0.0] * 64

    v_spaces = deterministic_vectorize("   \n\t  ", dim=64)
    assert v_spaces == [0.0] * 64


def test_deterministic_vectorize_custom_dimensions() -> None:
    """Valida o suporte a diferentes dimensões e rejeição de dimensões inválidas."""
    v64 = deterministic_vectorize("teste", dim=64)
    assert len(v64) == 64

    with pytest.raises(ValueError, match="positivo"):
        deterministic_vectorize("teste", dim=0)

    with pytest.raises(ValueError, match="positivo"):
        deterministic_vectorize("teste", dim=-5)


def test_cosine_similarity_edge_cases() -> None:
    """Testa similaridade de cosseno com vetores idênticos, ortogonais e nulos."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    zeros = [0.0, 0.0, 0.0]

    assert math.isclose(cosine_similarity(v1, v2), 1.0, rel_tol=1e-5)
    assert math.isclose(cosine_similarity(v1, v3), 0.0, abs_tol=1e-5)
    assert cosine_similarity(v1, zeros) == 0.0
    assert cosine_similarity(v1, [1.0, 0.0]) == 0.0  # Dimensões diferentes
    assert cosine_similarity([], v1) == 0.0


# =====================================================================
# 2. Testes de Armazenamento e Recuperação Híbrida (JarvisMemoryStore)
# =====================================================================


def test_jarvis_memory_store_add_and_search_in_memory() -> None:
    """Testa adição e busca de memórias combinando cosseno e overlap lexical."""
    store = JarvisMemoryStore(":memory:")

    m1 = store.add_memory(
        topic="autenticacao",
        content="Implementação de login seguro com tokens JWT e OAuth2",
    )
    m2 = store.add_memory(
        topic="banco_dados",
        content="Configuração do pool de conexões SQLite em modo WAL",
    )
    m3 = store.add_memory(
        topic="frontend",
        content="Componentes estilizados com TailwindCSS e animações CSS",
    )

    assert isinstance(m1, MemoryEntry)
    assert m1.entry_id
    assert m1.topic == "autenticacao"

    # Busca com termos fortemente alinhados à memória m1
    results = store.search_memories("login seguro com token OAuth2", top_k=2)
    assert len(results) == 2
    top = results[0]
    assert top.topic == "autenticacao"
    assert top.score > 0.4
    assert "JWT" in top.content

    # Busca com top_k=1
    single = store.search_memories("SQLite WAL", top_k=1)
    assert len(single) == 1
    assert single[0].topic == "banco_dados"

    store.close()


def test_jarvis_memory_store_file_persistence(tmp_path: Path) -> None:
    """Garante que os dados persistem em arquivo SQLite no disco."""
    db_file = tmp_path / "jarvis_test.db"

    with JarvisMemoryStore(db_file) as store1:
        store1.add_memory(topic="deploy", content="Deploy automatizado via Docker")

    with JarvisMemoryStore(db_file) as store2:
        results = store2.search_memories("deploy Docker", top_k=5)
        assert len(results) == 1
        assert results[0].topic == "deploy"
        assert "Docker" in results[0].content


def test_jarvis_memory_store_task_queue() -> None:
    """Testa enfileiramento, listagem e conclusão de tarefas no SQLite."""
    store = JarvisMemoryStore(":memory:")

    # Enfileirar tarefas
    t1 = store.enqueue_task(
        prompt="Gerar relatório de vendas",
        intent="analytics",
    )
    t2 = store.enqueue_task(
        prompt="Auditar vulnerabilidades de dependências",
        intent="security",
    )

    assert isinstance(t1, TaskItem)
    assert t1.status == "pending"
    assert t1.result is None

    # Listar tarefas
    tasks = store.list_tasks(limit=10)
    assert len(tasks) == 2
    task_ids = [t.task_id for t in tasks]
    assert t1.task_id in task_ids
    assert t2.task_id in task_ids

    # Concluir tarefa t1
    store.complete_task(
        task_id=t1.task_id,
        status="completed",
        result="Relatório salvo em relatorio_vendas.pdf",
    )

    updated_tasks = store.list_tasks(limit=10)
    completed_t1 = next(t for t in updated_tasks if t.task_id == t1.task_id)
    assert completed_t1.status == "completed"
    assert completed_t1.result == "Relatório salvo em relatorio_vendas.pdf"

    store.close()


# =====================================================================
# 3. Testes do Roteador Multi-LLM (MultiLLMRouter)
# =====================================================================


def test_router_deterministic_local_fallback_no_keys() -> None:
    """Verifica se router sem chaves vai direto para o fallback local resumindo o contexto."""
    router = MultiLLMRouter(
        gemini_api_key=None,
        openrouter_api_key=None,
        model="gemini-2.5-flash",
    )

    # Chamada com contexto
    resp_with_ctx = asyncio.run(
        router.generate(
            prompt="Escreva um resumo executivo",
            context="Reunião com diretoria agendada para 15:00 sobre orçamento",
        )
    )
    assert isinstance(resp_with_ctx, LLMResponse)
    assert resp_with_ctx.provider == "local_fallback"
    assert resp_with_ctx.model == "gemini-2.5-flash"
    assert "Reunião com diretoria" in resp_with_ctx.text
    assert "Escreva um resumo" in resp_with_ctx.text

    # Chamada sem contexto
    resp_no_ctx = asyncio.run(
        router.generate(prompt="Qual a hora atual?")
    )
    assert resp_no_ctx.provider == "local_fallback"
    assert "Qual a hora atual?" in resp_no_ctx.text


def test_router_gemini_success_mock() -> None:
    """Testa geração bem-sucedida via Gemini REST API."""
    def gemini_handler(request: httpx.Request) -> httpx.Response:
        assert "generativelanguage.googleapis.com" in str(request.url)
        body = json.loads(request.read())
        assert "contents" in body
        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Resposta gerada pelo Google Gemini 2.5 Flash."}
                        ]
                    }
                }
            ]
        }
        return httpx.Response(200, json=response_payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler))
    router = MultiLLMRouter(
        gemini_api_key="valid-gemini-key",
        openrouter_api_key="openrouter-key",
        model="gemini-2.5-flash",
        http_client=client,
    )

    resp = asyncio.run(router.generate("Olá Jarvis", context="Memória prévia"))
    assert resp.provider == "gemini"
    assert "Google Gemini" in resp.text
    assert resp.model == "gemini-2.5-flash"


def test_router_failover_gemini_to_openrouter() -> None:
    """Testa o failover transparente quando o Gemini falha e o OpenRouter responde com sucesso."""
    call_log: list[str] = []

    def failover_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "generativelanguage.googleapis.com" in url_str:
            call_log.append("gemini")
            # Simula erro de cota ou indisponibilidade no Gemini
            return httpx.Response(503, json={"error": "Service Unavailable"})
        elif "openrouter.ai" in url_str:
            call_log.append("openrouter")
            assert "Bearer test-openrouter-key" in request.headers.get("authorization", "")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "Resposta atendida pelo OpenRouter fallback."
                            }
                        }
                    ]
                },
            )
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(failover_handler))
    router = MultiLLMRouter(
        gemini_api_key="failing-gemini-key",
        openrouter_api_key="test-openrouter-key",
        model="gemini-2.5-flash",
        http_client=client,
    )

    resp = asyncio.run(
        router.generate(
            prompt="Execute a rotina",
            context="Ambiente de produção",
        )
    )

    assert call_log == ["gemini", "openrouter"]
    assert resp.provider == "openrouter"
    assert "OpenRouter fallback" in resp.text


def test_router_failover_all_to_local_fallback() -> None:
    """Testa o caso em que Gemini e OpenRouter falham (status 500/502), caindo no fallback local."""
    def all_fail_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "generativelanguage.googleapis.com" in url_str:
            return httpx.Response(500, json={"error": "Gemini Internal Server Error"})
        elif "openrouter.ai" in url_str:
            return httpx.Response(502, json={"error": "Bad Gateway"})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(all_fail_handler))
    router = MultiLLMRouter(
        gemini_api_key="key-gemini",
        openrouter_api_key="key-openrouter",
        model="gemini-2.5-flash",
        http_client=client,
    )

    resp = asyncio.run(
        router.generate(
            prompt="Gerar token de acesso",
            context="Usuário admin ID 42",
        )
    )

    assert resp.provider == "local_fallback"
    assert "Usuário admin ID 42" in resp.text
    assert "Gerar token de acesso" in resp.text


def test_router_ephemeral_client_mock_post() -> None:
    """Verifica funcionamento sem injeção explícita de http_client, mockando httpx.AsyncClient.post."""
    router = MultiLLMRouter(
        gemini_api_key="test-gemini-key",
        openrouter_api_key=None,
        model="gemini-2.5-flash",
    )

    mock_gemini_resp = httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Resposta via post mockado."}]
                    }
                }
            ]
        },
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_gemini_resp
        resp = asyncio.run(router.generate("Prompt de teste"))
        assert resp.provider == "gemini"
        assert resp.text == "Resposta via post mockado."
        assert mock_post.called
