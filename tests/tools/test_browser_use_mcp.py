from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import ClientSession
from mcp.types import ImageContent, ListToolsResult, TextContent, Tool


_CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.toml"
_CREATED_TEST_CONFIG = not _CONFIG_PATH.exists()
if _CREATED_TEST_CONFIG:
    _CONFIG_PATH.write_text(
        '[llm]\nmodel = "test"\nbase_url = "http://localhost"\napi_key = "test"\n'
        '\n[daytona]\ndaytona_api_key = "test"\n'
    )

try:
    from app.agent import manus as manus_module
    from app.tool.mcp import MCPClients, MCPClientTool
finally:
    if _CREATED_TEST_CONFIG:
        _CONFIG_PATH.unlink()


class FakeSession(ClientSession):
    def __init__(self, *, instructions="", content=None):
        self.instructions = instructions
        self.content = content or []

    async def initialize(self):
        return SimpleNamespace(instructions=self.instructions)

    async def list_tools(self):
        return ListToolsResult(
            tools=[
                Tool(
                    name="browser_exec",
                    description="Execute Browser Use CLI 3.0 code",
                    inputSchema={"type": "object"},
                ),
                Tool(
                    name="browser_screenshot",
                    description="Capture the current page",
                    inputSchema={"type": "object"},
                ),
            ]
        )

    async def call_tool(self, name, arguments):
        return SimpleNamespace(content=self.content)


@pytest.mark.asyncio
async def test_mcp_preserves_server_instructions_and_native_tool_names():
    clients = MCPClients()
    clients.sessions["browser_use"] = FakeSession(instructions="canonical skill")

    await clients._initialize_and_list_tools("browser_use", tool_name_prefix=False)

    assert clients.server_instructions["browser_use"] == "canonical skill"
    assert set(clients.tool_map) == {"browser_exec", "browser_screenshot"}


@pytest.mark.asyncio
async def test_mcp_forwards_text_and_screenshot_content():
    session = FakeSession(
        content=[
            TextContent(type="text", text="done"),
            ImageContent(type="image", data="cG5n", mimeType="image/png"),
        ]
    )
    tool = MCPClientTool(
        name="browser_screenshot",
        description="Capture the current page",
        parameters={"type": "object"},
        session=session,
        server_id="browser_use",
        original_name="browser_screenshot",
    )

    result = await tool.execute()

    assert result.output == "done"
    assert result.base64_image == "cG5n"


@pytest.mark.asyncio
async def test_manus_enables_cli_mcp_by_default(monkeypatch):
    calls = []

    async def record_connection(self, *args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.delenv("OPENMANUS_DISABLE_BROWSER_USE", raising=False)
    monkeypatch.setattr(manus_module.config.mcp_config, "servers", {})
    monkeypatch.setattr(manus_module.Manus, "connect_mcp_server", record_connection)

    await manus_module.Manus.model_construct().initialize_mcp_servers()

    assert calls == [
        (
            ("uvx", "browser_use"),
            {
                "use_stdio": True,
                "stdio_args": ["browser-use", "--cli-mcp"],
                "tool_name_prefix": False,
            },
        )
    ]
