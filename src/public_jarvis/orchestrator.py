"""Orquestrador assíncrono de tarefas e agentes para o Public-Jarvis.

Integra classificação de intenção, autorização de usuários, execução segura de comandos,
busca semântica em memória episódica, roteamento multi-LLM e paginação de respostas.
"""

from __future__ import annotations

import re
from typing import Literal
from pydantic import BaseModel, Field

from public_jarvis.config import JarvisSettings
from public_jarvis.memory import JarvisMemoryStore
from public_jarvis.router import MultiLLMRouter
from public_jarvis.security import SafeShellError, is_user_authorized, run_safe

IntentType = Literal["shell", "code", "analysis", "memory", "general"]


class AgentContext(BaseModel):
    """Contexto de execução do agente."""

    user_id: int
    workspace: str
    is_authorized: bool = False


class OrchestratorResult(BaseModel):
    """Resultado estruturado gerado pelo orquestrador."""

    intent: str
    response: str
    provider: str
    task_id: str
    pages: list[str] = Field(default_factory=list)


# Prefixos e comandos explícitos de shell
_SHELL_PREFIXES = ("!", "$", "sh ", "bash ", "run ", "exec ")
_SHELL_COMMANDS = (
    "ls", "pwd", "df", "free", "ps", "uptime", "git", "python3", "cat",
    "echo", "date", "whoami", "uname", "du", "head", "tail", "wc", "grep", "find",
    "rm", "sudo", "kill", "chmod", "chown", "mv", "cp", "mkdir", "rmdir", "touch",
    "nano", "vim", "curl", "wget", "bash", "sh", "zsh", "sed", "awk", "systemctl",
    "service", "apt", "dnf", "yum", "pip", "npm", "pkill", "top", "htop", "nc", "dd",
)
_SHELL_OPERATORS = (";", "&&", "||", "|", "$(", "`", ">", "<")

# Palavras-chave para classificação
_MEMORY_KEYWORDS = (
    "lembrar", "lembre", "lembre-se", "guardar", "guarde", "memória", "memoria",
    "memorizar", "salvar memória", "anotar", "anote", "esquecer", "esqueça"
)

_CODE_KEYWORDS = (
    "código", "codigo", "script", "função", "funcao", "refatorar", "refatore",
    "debug", "def ", "class ", "bug", "implementar", "implemente", "programar",
    "code", "syntax", "algoritmo", "unit test", "teste unitário", "pytest"
)

_ANALYSIS_KEYWORDS = (
    "analisar", "analise", "análise", "diagnóstico", "diagnostico", "diagnosticar",
    "investigar", "investigue", "auditar", "auditoria", "audite", "métrica", "metrica",
    "métricas", "metricas", "log", "logs", "performance", "cpu", "ram", "benchmark",
    "relatório", "relatorio", "inspecionar", "inspecione"
)


def classify_intent(prompt: str) -> IntentType:
    """Classifica a intenção do prompt em shell, code, analysis, memory ou general."""
    if not prompt or not prompt.strip():
        return "general"

    clean_prompt = prompt.strip()
    lower_prompt = clean_prompt.lower()

    # 1. Shell explícito por prefixo
    for prefix in _SHELL_PREFIXES:
        if lower_prompt.startswith(prefix):
            return "shell"

    # 2. Shell por operadores de comando
    for op in _SHELL_OPERATORS:
        if op in clean_prompt:
            return "shell"

    # 3. Shell explícito por primeiro token sendo comando do sistema
    first_token = re.split(r"\s+", clean_prompt)[0].lower()
    if first_token in _SHELL_COMMANDS:
        return "shell"

    # 4. Memória (evitar falso positivo se referindo a 'memória ram' em análise)
    if "memória ram" not in lower_prompt and "memoria ram" not in lower_prompt:
        for kw in _MEMORY_KEYWORDS:
            if kw in lower_prompt:
                return "memory"

    # 5. Análise
    for kw in _ANALYSIS_KEYWORDS:
        if kw in lower_prompt:
            return "analysis"

    # 6. Código
    for kw in _CODE_KEYWORDS:
        if kw in lower_prompt:
            return "code"

    return "general"


def paginate_text(text: str, max_chars: int = 3500) -> list[str]:
    """Divide um texto em páginas respeitando o limite máximo de caracteres por página.

    Tenta quebrar em limites naturais (parágrafos, linhas, frases ou palavras).
    """
    if max_chars <= 0:
        raise ValueError("max_chars deve ser maior que zero")

    if not text:
        return [""]

    if len(text) <= max_chars:
        return [text]

    pages: list[str] = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= max_chars:
            pages.append(remaining)
            break

        # Tentar quebra por parágrafo
        split_idx = remaining.rfind("\n\n", 0, max_chars)
        if split_idx == -1 or split_idx < max_chars // 3:
            # Tentar quebra por linha simples
            split_idx = remaining.rfind("\n", 0, max_chars)
        if split_idx == -1 or split_idx < max_chars // 3:
            # Tentar quebra por espaço de palavra
            split_idx = remaining.rfind(" ", 0, max_chars)
        if split_idx == -1 or split_idx < max_chars // 3:
            # Corte rígido se não houver divisor razoável
            split_idx = max_chars

        chunk = remaining[:split_idx].rstrip()
        if chunk:
            pages.append(chunk)

        remaining = remaining[split_idx:].lstrip("\r\n ")

    return pages if pages else [text]


class JarvisOrchestrator:
    """Orquestrador central do Public-Jarvis."""

    def __init__(
        self,
        settings: JarvisSettings,
        memory: JarvisMemoryStore,
        router: MultiLLMRouter,
    ):
        self.settings = settings
        self.memory = memory
        self.router = router

    async def handle(self, ctx: AgentContext, prompt: str) -> OrchestratorResult:
        """Processa uma solicitação de forma assíncrona com segurança, memória e roteamento."""
        clean_prompt = prompt.strip()

        # 1. Validação de autorização
        if self.settings.allowed_users:
            if not is_user_authorized(ctx.user_id, self.settings.allowed_users):
                ctx.is_authorized = False
                task = self.memory.enqueue_task(prompt=clean_prompt, intent="general")
                denied_msg = f"Acesso negado: Usuário {ctx.user_id} não autorizado."
                self.memory.complete_task(
                    task_id=task.task_id,
                    status="rejected",
                    result=denied_msg,
                )
                return OrchestratorResult(
                    intent="general",
                    response=denied_msg,
                    provider="system",
                    task_id=task.task_id,
                    pages=paginate_text(denied_msg),
                )

        ctx.is_authorized = True

        # 2. Classificação de intenção
        intent = classify_intent(clean_prompt)

        # 3. Registro da tarefa na fila
        task = self.memory.enqueue_task(prompt=clean_prompt, intent=intent)

        # 4. Execução baseada na intenção
        if intent == "shell":
            cmd_to_run = clean_prompt
            # Normalizar prefixos de shell
            if cmd_to_run.startswith("!"):
                cmd_to_run = cmd_to_run[1:].strip()
            elif cmd_to_run.startswith("$ "):
                cmd_to_run = cmd_to_run[2:].strip()
            elif cmd_to_run.lower().startswith("run "):
                cmd_to_run = cmd_to_run[4:].strip()
            elif cmd_to_run.lower().startswith("exec "):
                cmd_to_run = cmd_to_run[5:].strip()

            try:
                retcode, stdout, stderr = run_safe(cmd_to_run, cwd=ctx.workspace)
                if retcode == 0:
                    response_text = (
                        stdout.strip()
                        if stdout.strip()
                        else "(comando executado com sucesso, sem saída padrão)"
                    )
                    status = "completed"
                    provider = "shell"
                else:
                    parts = []
                    if stdout.strip():
                        parts.append(f"Saída:\n{stdout.strip()}")
                    if stderr.strip():
                        parts.append(f"Erro:\n{stderr.strip()}")
                    err_msg = "\n".join(parts) or f"Comando encerrou com código {retcode}"
                    response_text = f"Falha na execução (código {retcode}):\n{err_msg}"
                    status = "failed"
                    provider = "shell"
            except SafeShellError as exc:
                response_text = f"Bloqueio de segurança SafeShell: {exc}"
                status = "failed"
                provider = "safe_shell"
            except Exception as exc:
                response_text = f"Erro na execução do comando: {exc}"
                status = "failed"
                provider = "shell"

            self.memory.complete_task(
                task_id=task.task_id,
                status=status,
                result=response_text,
            )
        else:
            # 5. Busca de memórias relevantes
            memories = self.memory.search_memories(query=clean_prompt, top_k=5)
            context_parts: list[str] = []
            for mem in memories:
                if mem.score > 0.05 or len(context_parts) < 2:
                    context_parts.append(f"[{mem.topic}] {mem.content}")

            context_str = "\n".join(context_parts)

            llm_resp = await self.router.generate(prompt=clean_prompt, context=context_str)
            response_text = llm_resp.text
            provider = llm_resp.provider

            self.memory.complete_task(
                task_id=task.task_id,
                status="completed",
                result=response_text,
            )

        # 6. Paginação do resultado
        pages = paginate_text(response_text, max_chars=3500)

        return OrchestratorResult(
            intent=intent,
            response=response_text,
            provider=provider,
            task_id=task.task_id,
            pages=pages,
        )
