"""Módulo de segurança e execução segura de comandos (prevenção CWE-78)."""

from __future__ import annotations

import shlex
import subprocess
from typing import Final

ALLOWED_BINARIES: Final[frozenset[str]] = frozenset({
    "ls",
    "pwd",
    "df",
    "free",
    "ps",
    "uptime",
    "git",
    "python3",
    "cat",
    "echo",
    "date",
    "whoami",
    "uname",
    "du",
    "head",
    "tail",
    "wc",
    "grep",
    "find",
})

ALLOWED_SUBCOMMANDS: Final[dict[str, frozenset[str]]] = {
    "git": frozenset({"status", "log", "diff", "show", "branch"}),
    "python3": frozenset({"--version", "-c", "-m"}),
}

FORBIDDEN_OPERATORS: Final[tuple[str, ...]] = (";", "|", "&&", "$(", "`", "\n", "\r")


class SafeShellError(Exception):
    """Exceção levantada para comandos rejeitados por políticas de segurança."""


def validate_command(cmd: str) -> list[str]:
    """Valida comando contra políticas de segurança e retorna lista de argumentos.

    Bloqueia:
    - Comandos vazios ou nulos;
    - Operadores perigosos de encadeamento/injeção (';', '|', '&&', '$()', backticks, quebras de linha);
    - Caminhos absolutos ou relativos no binário;
    - Binários fora da whitelist ALLOWED_BINARIES;
    - Subcomandos fora da whitelist ALLOWED_SUBCOMMANDS (para git e python3).
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise SafeShellError("Comando não pode ser vazio.")

    for op in FORBIDDEN_OPERATORS:
        if op in cmd:
            raise SafeShellError(f"Operador ou caractere de injeção proibido detectado: {op!r}")

    try:
        tokens = shlex.split(cmd)
    except ValueError as exc:
        raise SafeShellError(f"Erro de sintaxe no comando: {exc}") from exc

    if not tokens:
        raise SafeShellError("Comando não pode ser vazio após análise léxica.")

    binary = tokens[0]

    if "/" in binary or "\\" in binary:
        raise SafeShellError(f"Execução de binários por caminho não é permitida: {binary}")

    if binary not in ALLOWED_BINARIES:
        raise SafeShellError(f"Binário não autorizado: {binary}")

    if binary in ALLOWED_SUBCOMMANDS:
        allowed_subs = ALLOWED_SUBCOMMANDS[binary]
        if len(tokens) < 2:
            raise SafeShellError(f"Subcomando ou opção obrigatória para o binário '{binary}'.")

        subcmd = tokens[1]
        if subcmd not in allowed_subs:
            raise SafeShellError(
                f"Subcomando/opção '{subcmd}' não permitida para '{binary}'. "
                f"Permitidos: {sorted(allowed_subs)}"
            )

    return tokens


def run_safe(cmd: str, timeout: int = 30, cwd: str | None = None) -> tuple[int, str, str]:
    """Executa um comando de forma estritamente segura (sem shell=True, CWE-78).

    Retorna: (returncode, stdout, stderr)
    Lança: SafeShellError se validação falhar, estourar timeout ou houver falha de execução.
    """
    args = validate_command(cmd)

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            shell=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        raise SafeShellError(f"Comando excedeu o tempo limite de {timeout}s: {cmd}") from exc
    except FileNotFoundError as exc:
        raise SafeShellError(f"Executável não encontrado no sistema: {args[0]}") from exc
    except OSError as exc:
        raise SafeShellError(f"Erro de sistema operacional ao executar {args[0]}: {exc}") from exc


def is_user_authorized(user_id: int, allowed_users: list[int]) -> bool:
    """Verifica se o user_id pertence à lista de usuários autorizados."""
    if not allowed_users:
        return False
    try:
        return int(user_id) in allowed_users
    except (ValueError, TypeError):
        return False
