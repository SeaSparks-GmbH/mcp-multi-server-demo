from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
from mcp.types import CallToolResult, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount

import uvicorn

# Create the server
server = Server("MathServer")

def add(a: int, b: int) -> int:
    """Adds two integers.

    Args:
         a: The first integer.
         b: The second integer.
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies two integers.

    Args:
         a: The first integer.
         b: The second integer.
    """
    return a * b

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="add",
            description="Adds two integers",
            inputSchema={
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
            },
        ),
        types.Tool(
            name="multiply",
            description="Multiplies two integers",
            inputSchema={
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
            },
        )
    ]

@server.call_tool()
async def handle_call(func_name: str, args: dict):
    if func_name not in ["add", "multiply"]:
        raise ValueError("Unknown tool")
    func = globals().get(func_name)
    if callable(func):
        result = func(**args)
        return_value = TextContent(type="text", text=str(result))
    else:
        return_value = TextContent(type="text", text=(f"Function {func_name} not defined."))
    return [return_value]

# SSE transport setup
sse_transport = SseServerTransport("/messages/")

# Route to handle /sse (connects and runs the server)
async def sse_handler(request):
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())

routes = [
    Route("/sse", endpoint=sse_handler),
    Mount("/messages/", app=sse_transport.handle_post_message),
]

starlette_app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=58000)