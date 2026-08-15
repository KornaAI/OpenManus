"""Browser automation through the isolated Browser Use CLI 3.0 runtime."""

import asyncio
import base64
import json
import os
import re
import shutil
from pathlib import Path
from typing import Generic, Optional, TypeVar

from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.tool.base import BaseTool, ToolResult


_DEFAULT_TIMEOUT_SECONDS = 300
_MIN_TIMEOUT_SECONDS = 5
_MAX_TIMEOUT_SECONDS = 1800
_OUTPUT_LIMIT = 20_000
_IMAGE_PATH_RE = re.compile(
    r"((?:[A-Za-z]:[\\/]|/)[^\s\"']+?\.(?:png|jpe?g|webp))", re.IGNORECASE
)

_BROWSER_DESCRIPTION = """\
Control a persistent browser with Browser Use CLI 3.0 (browser-harness).
Pass Python in `code`; browser helpers are already imported. Print values you
need in the result. Browser state persists across calls.

Core helpers:
- new_tab(url), goto_url(url), page_info(), list_tabs(), switch_tab(target)
- click_at_xy(x, y), type_text(text), fill_input(selector, text), press_key(key)
- js(code), cdp(method, ...), wait_for_load(), wait_for_element(selector)
- capture_screenshot(), scroll(x, y), close_tab(target)

Example:
  code="new_tab('https://example.com')\nwait_for_load()\nprint(page_info())"

Use accessibility-tree or DOM inspection before coordinates when possible.
"""

Context = TypeVar("Context")


def _browser_use_command() -> Optional[list[str]]:
    """Return an isolated Browser Use CLI command without importing it here."""
    override = os.environ.get("BROWSER_USE_BIN", "").strip()
    if override:
        return [str(Path(override).expanduser())]

    # uvx keeps Browser Use's fast-moving dependency set isolated from
    # OpenManus's own OpenAI and Pydantic pins.
    if uvx := shutil.which("uvx"):
        return [uvx, "browser-use"]

    if direct := shutil.which("browser-use"):
        return [direct]

    local_bin = Path.home() / ".local" / "bin" / "browser-use"
    if local_bin.is_file():
        return [str(local_bin)]
    return None


def _browser_env() -> dict[str, str]:
    env = dict(os.environ)
    # uvx/browser-use runs under its own interpreter. Parent Python import
    # paths can load ABI-incompatible packages such as pydantic_core.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.setdefault("ANONYMIZED_TELEMETRY", "false")
    env.setdefault("BH_CLIENT", "openmanus")
    env.setdefault("BH_AGENT_WORKSPACE", str(config.workspace_root))

    browser_config = config.browser_config
    if not browser_config:
        return env

    if browser_config.chrome_instance_path:
        env.setdefault("BH_CHROME_PATH", browser_config.chrome_instance_path)

    if not (env.get("BU_CDP_URL") or env.get("BU_CDP_WS")):
        endpoint = browser_config.cdp_url or browser_config.wss_url
        if endpoint:
            key = (
                "BU_CDP_URL"
                if endpoint.startswith(("http://", "https://"))
                else "BU_CDP_WS"
            )
            env[key] = endpoint
        elif browser_config.headless and env.get("BROWSER_USE_API_KEY"):
            # The CLI controls visible local Chrome. Preserve headless server
            # behavior by opting into Browser Use Cloud when credentials exist.
            env.setdefault("BU_AUTOSPAWN", "1")
    return env


def _bounded_output(value: str) -> str:
    if len(value) <= _OUTPUT_LIMIT:
        return value
    return f"[output truncated to last {_OUTPUT_LIMIT} characters]\n{value[-_OUTPUT_LIMIT:]}"


def _screenshot_data(stdout: str) -> Optional[str]:
    for raw_path in reversed(_IMAGE_PATH_RE.findall(stdout)):
        path = Path(raw_path)
        try:
            if path.is_file() and path.stat().st_size <= 20 * 1024 * 1024:
                return base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
    return None


class BrowserUseTool(BaseTool, Generic[Context]):
    """One compact tool over Browser Use's persistent CDP harness."""

    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python using the pre-imported Browser Use CLI helpers.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": _MIN_TIMEOUT_SECONDS,
                "maximum": _MAX_TIMEOUT_SECONDS,
                "description": "Execution timeout. Defaults to 300 seconds.",
            },
        },
        "required": ["code"],
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, value: dict, info: ValidationInfo) -> dict:
        if not value:
            raise ValueError("Parameters cannot be empty")
        return value

    async def execute(
        self,
        code: str,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs,
    ) -> ToolResult:
        if not code or not code.strip():
            return ToolResult(error="Browser Use code cannot be empty")
        if not _MIN_TIMEOUT_SECONDS <= timeout_seconds <= _MAX_TIMEOUT_SECONDS:
            return ToolResult(
                error=(
                    f"timeout_seconds must be between {_MIN_TIMEOUT_SECONDS} "
                    f"and {_MAX_TIMEOUT_SECONDS}"
                )
            )

        command = _browser_use_command()
        if command is None:
            return ToolResult(
                error=(
                    "Browser Use CLI is unavailable. Install uv, then run "
                    "`uv tool install browser-use`."
                )
            )

        async with self.lock:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=_browser_env(),
                )
            except OSError as exc:
                return ToolResult(error=f"Failed to start Browser Use CLI: {exc}")

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(code.encode("utf-8")), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return ToolResult(
                    error=f"Browser Use CLI timed out after {timeout_seconds} seconds"
                )

        stdout = _bounded_output(stdout_bytes.decode("utf-8", errors="replace").strip())
        stderr = _bounded_output(stderr_bytes.decode("utf-8", errors="replace").strip())
        if process.returncode != 0:
            detail = stderr or stdout or "No error output"
            return ToolResult(
                output=stdout or None,
                error=f"Browser Use CLI exited with code {process.returncode}: {detail}",
            )

        return ToolResult(
            output=stdout or "Browser command completed successfully.",
            base64_image=_screenshot_data(stdout),
        )

    async def get_current_state(self) -> ToolResult:
        """Return page_info as JSON for BrowserAgent's existing state helper."""
        result = await self.execute(
            "import json\n"
            "_state = page_info()\n"
            "if hasattr(_state, 'model_dump'):\n"
            "    _state = _state.model_dump()\n"
            "print(json.dumps(_state, default=str))"
        )
        if result.error:
            return result

        # Keep only the final JSON line in case the CLI emitted startup text.
        for line in reversed(str(result.output).splitlines()):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            return ToolResult(
                output=json.dumps(parsed), base64_image=result.base64_image
            )
        return result

    async def cleanup(self) -> None:
        """The CLI daemon owns browser lifecycle and persists between calls."""

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        tool = cls()
        tool.tool_context = context
        return tool
