"""Configurações e carregamento de ambiente para o Public-Jarvis."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field
import dotenv

_PLACEHOLDER_PREFIX = "your_"
_PLACEHOLDER_SUFFIX = "_here"


def _is_placeholder(val: str | None) -> bool:
    """Verifica se uma string é vazia ou representa um valor placeholder."""
    if val is None:
        return True
    s = val.strip()
    if not s:
        return True
    lower = s.lower()
    if lower.startswith(_PLACEHOLDER_PREFIX) and lower.endswith(_PLACEHOLDER_SUFFIX):
        return True
    if lower.startswith("your_"):
        return True
    return False


def _clean_str(val: str | None) -> str | None:
    """Retorna valor limpo ou None se for placeholder ou vazio."""
    if _is_placeholder(val):
        return None
    assert val is not None
    return val.strip()


def _parse_allowed_users(raw: str | None) -> list[int]:
    """Faz parse de lista CSV de inteiros ignorando placeholders e valores inválidos."""
    if _is_placeholder(raw):
        return []
    assert raw is not None
    users: list[int] = []
    for item in raw.split(","):
        item_str = item.strip()
        if not item_str:
            continue
        try:
            users.append(int(item_str))
        except ValueError:
            continue
    return users


class JarvisSettings(BaseModel):
    """Configurações centrais do Public-Jarvis."""

    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    telegram_bot_token: str | None = None
    allowed_users: list[int] = Field(default_factory=list)
    db_path: Path = Path("data/jarvis_memory.sqlite")
    model: str = "gemini-2.5-flash"


def load_settings(env_path: Path | None = None) -> JarvisSettings:
    """Lê variáveis de ambiente e instancia JarvisSettings.

    - Se env_path for informado e existir, carrega via dotenv.load_dotenv(override=True).
    - Se env_path for None, executa dotenv.load_dotenv() padrão.
    - Ignora valores placeholder ('your_*_here', vazios).
    - Faz parse de ALLOWED_USERS como CSV de inteiros.
    """
    if env_path is not None:
        p = Path(env_path)
        if p.exists():
            dotenv.load_dotenv(dotenv_path=p, override=True)
    else:
        dotenv.load_dotenv(override=False)

    gemini_key = _clean_str(os.getenv("GEMINI_API_KEY"))
    openrouter_key = _clean_str(os.getenv("OPENROUTER_API_KEY"))
    telegram_token = _clean_str(os.getenv("TELEGRAM_BOT_TOKEN"))
    allowed_users = _parse_allowed_users(os.getenv("ALLOWED_USERS"))

    db_env = _clean_str(os.getenv("JARVIS_DB_PATH") or os.getenv("DB_PATH"))
    db_path = Path(db_env) if db_env else Path("data/jarvis_memory.sqlite")

    model_env = _clean_str(os.getenv("JARVIS_MODEL") or os.getenv("MODEL"))
    model = model_env if model_env else "gemini-2.5-flash"

    return JarvisSettings(
        gemini_api_key=gemini_key,
        openrouter_api_key=openrouter_key,
        telegram_bot_token=telegram_token,
        allowed_users=allowed_users,
        db_path=db_path,
        model=model,
    )
