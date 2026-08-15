import json
import sys
from pathlib import Path

import pytest


# OpenManus initializes its global config during imports and expects users to
# have copied config.example.toml. Keep this unit test hermetic in a fresh clone.
_CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.toml"
_CREATED_TEST_CONFIG = not _CONFIG_PATH.exists()
if _CREATED_TEST_CONFIG:
    _CONFIG_PATH.write_text(
        '[llm]\nmodel = "test"\nbase_url = "http://localhost"\napi_key = "test"\n'
        "\n[browser]\nheadless = true\n"
        '\n[daytona]\ndaytona_api_key = "test"\n'
    )

try:
    from app.tool import browser_use_tool
    from app.tool.browser_use_tool import BrowserUseTool
finally:
    if _CREATED_TEST_CONFIG:
        _CONFIG_PATH.unlink()


def _fake_cli(tmp_path, body: str):
    script = tmp_path / "fake_browser_use.py"
    script.write_text(body)
    return [sys.executable, str(script)]


@pytest.mark.asyncio
async def test_executes_code_over_stdin(tmp_path, monkeypatch):
    command = _fake_cli(
        tmp_path,
        "import sys\nprint('received:' + sys.stdin.read())\n",
    )
    monkeypatch.setattr(browser_use_tool, "_browser_use_command", lambda: command)

    result = await BrowserUseTool().execute(code="print(page_info())")

    assert result.error is None
    assert result.output == "received:print(page_info())"


@pytest.mark.asyncio
async def test_reports_nonzero_exit(tmp_path, monkeypatch):
    command = _fake_cli(
        tmp_path,
        "import sys\nsys.stdin.read()\nprint('bad helper', file=sys.stderr)\nraise SystemExit(3)\n",
    )
    monkeypatch.setattr(browser_use_tool, "_browser_use_command", lambda: command)

    result = await BrowserUseTool().execute(code="unknown_helper()")

    assert result.output is None
    assert "code 3" in result.error
    assert "bad helper" in result.error


@pytest.mark.asyncio
async def test_get_current_state_keeps_final_json_line(tmp_path, monkeypatch):
    command = _fake_cli(
        tmp_path,
        "import sys\nsys.stdin.read()\nprint('daemon started')\n"
        'print(\'{"url": "https://example.com", "title": "Example"}\')\n',
    )
    monkeypatch.setattr(browser_use_tool, "_browser_use_command", lambda: command)

    result = await BrowserUseTool().get_current_state()

    assert result.error is None
    assert json.loads(result.output) == {
        "url": "https://example.com",
        "title": "Example",
    }


def test_browser_env_isolates_python_and_maps_cdp(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/wrong/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/wrong/python")
    monkeypatch.delenv("BU_CDP_URL", raising=False)
    monkeypatch.delenv("BU_CDP_WS", raising=False)
    monkeypatch.setattr(
        browser_use_tool.config.browser_config,
        "cdp_url",
        "http://127.0.0.1:9222",
    )

    env = browser_use_tool._browser_env()

    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env["BU_CDP_URL"] == "http://127.0.0.1:9222"
    assert env["ANONYMIZED_TELEMETRY"] == "false"
    assert env["BH_CLIENT"] == "openmanus"
    assert env["BH_AGENT_WORKSPACE"].endswith("workspace")


def test_browser_env_maps_chrome_and_headless_cloud(monkeypatch):
    monkeypatch.delenv("BU_CDP_URL", raising=False)
    monkeypatch.delenv("BU_CDP_WS", raising=False)
    monkeypatch.delenv("BU_AUTOSPAWN", raising=False)
    monkeypatch.setenv("BROWSER_USE_API_KEY", "test-key")
    monkeypatch.setattr(browser_use_tool.config.browser_config, "cdp_url", None)
    monkeypatch.setattr(browser_use_tool.config.browser_config, "wss_url", None)
    monkeypatch.setattr(browser_use_tool.config.browser_config, "headless", True)
    monkeypatch.setattr(
        browser_use_tool.config.browser_config,
        "chrome_instance_path",
        "/opt/chrome",
    )

    env = browser_use_tool._browser_env()

    assert env["BH_CHROME_PATH"] == "/opt/chrome"
    assert env["BU_AUTOSPAWN"] == "1"


def test_uvx_runs_browser_use_in_isolated_environment(monkeypatch):
    monkeypatch.delenv("BROWSER_USE_BIN", raising=False)
    monkeypatch.setattr(
        browser_use_tool.shutil,
        "which",
        lambda name: "/usr/bin/uvx" if name == "uvx" else None,
    )

    assert browser_use_tool._browser_use_command() == [
        "/usr/bin/uvx",
        "browser-use",
    ]


def test_direct_binary_fallback(monkeypatch):
    monkeypatch.delenv("BROWSER_USE_BIN", raising=False)
    monkeypatch.setattr(
        browser_use_tool.shutil,
        "which",
        lambda name: "/usr/bin/browser-use" if name == "browser-use" else None,
    )
    assert browser_use_tool._browser_use_command() == ["/usr/bin/browser-use"]
