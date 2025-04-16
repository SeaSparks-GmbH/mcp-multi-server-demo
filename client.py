from contextlib import AsyncExitStack
from mcp.client.sse import sse_client
from mcp import ClientSession

class MultiServerClient:
    def __init__(self, endpoints):
        """
        Initialize with a dictionary of {server_name: sse_url}.
        Example:
        {
            "math_server": "http://127.0.0.1:58000/sse",
            "weather_server": "http://127.0.0.1:58001/sse"
        }
        """
        self.endpoints = endpoints
        self.sessions = {}
        self._exit_stack = AsyncExitStack()

    async def connect_all(self):
        for name, sse_url in self.endpoints.items():
            read, write = await self._exit_stack.enter_async_context(sse_client(sse_url))
            session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[name] = session

    async def disconnect_all(self):
        await self._exit_stack.aclose()

    async def list_all_tools(self):
        all_tools = {}
        for name, session in self.sessions.items():
            tools_response = await session.list_tools()
            all_tools[name] = tools_response.tools
        return all_tools

    async def call(self, server_name, tool_name, args):
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"No session for server '{server_name}'")
        return await session.call_tool(tool_name, args)