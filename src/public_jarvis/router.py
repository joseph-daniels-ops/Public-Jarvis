"""Roteador multi-LLM assíncrono para o Public-Jarvis.

Gerencia chamadas com failover automático:
1. Gemini REST API (se gemini_api_key disponível)
2. OpenRouter API (se openrouter_api_key disponível)
3. Local Fallback determinístico (resumo limpo do contexto sem quebrar a execução)
"""

from __future__ import annotations

import logging
from typing import Literal
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Resposta estruturada de geração de texto retornada pelo roteador."""

    text: str
    provider: Literal["gemini", "openrouter", "local_fallback"]
    model: str


class MultiLLMRouter:
    """Roteador inteligente de chamadas a modelos de linguagem com failover."""

    def __init__(
        self,
        gemini_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        http_client: httpx.AsyncClient | None = None,
    ):
        self.gemini_api_key = gemini_api_key
        self.openrouter_api_key = openrouter_api_key
        self.model = model
        self.http_client = http_client

    async def generate(self, prompt: str, context: str = "") -> LLMResponse:
        """Gera resposta consultando Gemini, OpenRouter ou fallback local em ordem."""
        has_gemini = bool(self.gemini_api_key and self.gemini_api_key.strip())
        has_openrouter = bool(
            self.openrouter_api_key and self.openrouter_api_key.strip()
        )

        if not has_gemini and not has_openrouter:
            return self._local_fallback(prompt, context)

        if self.http_client is not None:
            return await self._generate_with_client(self.http_client, prompt, context)

        async with httpx.AsyncClient(timeout=30.0) as client:
            return await self._generate_with_client(client, prompt, context)

    async def _generate_with_client(
        self, client: httpx.AsyncClient, prompt: str, context: str
    ) -> LLMResponse:
        """Executa o pipeline de tentativa e failover com o cliente HTTP fornecido."""
        # 1. Tentativa no Google Gemini REST API
        if self.gemini_api_key and self.gemini_api_key.strip():
            gemini_result = await self._call_gemini(client, prompt, context)
            if gemini_result is not None:
                return LLMResponse(
                    text=gemini_result,
                    provider="gemini",
                    model=self.model,
                )
            logger.warning("Falha ou indisponibilidade no Gemini API. Tentando OpenRouter...")

        # 2. Failover para OpenRouter API
        if self.openrouter_api_key and self.openrouter_api_key.strip():
            openrouter_result = await self._call_openrouter(client, prompt, context)
            if openrouter_result is not None:
                return LLMResponse(
                    text=openrouter_result,
                    provider="openrouter",
                    model=self.model,
                )
            logger.warning(
                "Falha ou indisponibilidade no OpenRouter API. Ativando fallback local..."
            )

        # 3. Fallback limpo local
        return self._local_fallback(prompt, context)

    async def _call_gemini(
        self, client: httpx.AsyncClient, prompt: str, context: str
    ) -> str | None:
        """Chama a API REST do Gemini para generateContent."""
        model_name = self.model.removeprefix("models/")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={self.gemini_api_key}"
        )
        full_prompt = (
            f"Contexto:\n{context}\n\nSolicitação:\n{prompt}"
            if context.strip()
            else prompt
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": full_prompt}
                    ]
                }
            ]
        }

        try:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates") or []
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
            logger.warning(
                "Gemini REST retornou status %s: %s",
                response.status_code,
                response.text,
            )
        except Exception as exc:
            logger.warning("Erro de conexão na requisição Gemini: %s", exc)
        return None

    async def _call_openrouter(
        self, client: httpx.AsyncClient, prompt: str, context: str
    ) -> str | None:
        """Chama a API do OpenRouter como camada de contingência."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/josedaniel/Public-Jarvis",
            "X-Title": "Public-Jarvis",
        }
        openrouter_model = self.model if "/" in self.model else f"google/{self.model}"
        full_prompt = (
            f"Contexto:\n{context}\n\nSolicitação:\n{prompt}"
            if context.strip()
            else prompt
        )
        payload = {
            "model": openrouter_model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
        }

        try:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if content:
                        return content
            logger.warning(
                "OpenRouter retornou status %s: %s",
                response.status_code,
                response.text,
            )
        except Exception as exc:
            logger.warning("Erro de conexão na requisição OpenRouter: %s", exc)
        return None

    def _local_fallback(self, prompt: str, context: str) -> LLMResponse:
        """Produz uma resposta offline limpa e informativa sem quebrar o fluxo."""
        if context.strip():
            text = (
                f"[Fallback local] Contexto recuperado:\n{context.strip()}\n\n"
                f"Solicitação: {prompt.strip()}"
            )
        else:
            text = f"[Fallback local] Solicitação recebida: {prompt.strip()}"

        return LLMResponse(
            text=text,
            provider="local_fallback",
            model=self.model,
        )
