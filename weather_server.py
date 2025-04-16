from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
from mcp.types import CallToolResult, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
import uvicorn

server = Server("WeatherServer")

@server.call_tool()
async def call_tool(name: str, arguments: dict)-> CallToolResult:
    if name == "get_weather":
        weather_report = get_weather(**arguments)

        return_value = TextContent(type="text", text=weather_report)

        return [return_value]

    raise ValueError("Unknown tool")

# 👇 Normal Python function — NOT decorated
def get_weather(city: str) -> str:
    """Get weather for a location.

        Args:
            city: name of location / city
        """
    return f"The weather in {city} is sunny and 25°C."

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="get_weather",
            description="Returns fake weather",
            inputSchema={
                "type": "object",
                "required": ["city"],
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City to get the weather for"
                    }
                }
            }
        )
    ]

sse = SseServerTransport("/messages/")

async def sse_handler(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())

routes = [
    Route("/sse", endpoint=sse_handler),
    Mount("/messages/", app=sse.handle_post_message),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=58001)
