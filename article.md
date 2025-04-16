# MCP From a Slightly Different Angle — How to Build A Local Multi-Server Systems Without FastMCP

## Introduction

Since Anthropic introduced the Model Context Protocol (MCP) in November 2024, a growing number of articles, tutorials, and deep-dives have explored its concepts and practical applications. Between the official documentation and examples ([Anthropic MCP page](https://modelcontextprotocol.io)) and the Python SDK ([Python SDK on GitHub](https://github.com/modelcontextprotocol/python-sdk)), there’s already a solid foundation available for practitioners and curious developers alike.

  


 Many MCP tutorials are based on [FastMCP](https://github.com/jlowin/fastmcp?tab=readme-ov-file), a very useful high-level wrapper that handles the complex protocol details and server management.

This article aims to complement those resources with a working, lower-level, multi-server example that exposes more of the machinery behind the scenes.

Rather than relying on FastMCP, we’ll show how to define, launch, and interact with multiple independent MCP servers — each potentially running on different ports or hosts — using the SSE transport protocol that plays a central role in MCP beyond local stdin/stdout setups.

## The MCP-Trinity — Servers, Client and Host

Large language models and the systems built around them — such as agentic frameworks or autonomous assistants — increasingly rely on diverse resources, tools, and services. These capabilities extend the model's functionality beyond its static training data and weights, helping it interact with a dynamic world. Some tools are rule-based, adding structure and reliability to the otherwise stochastic behavior of LLMs.

Such tools can be broadly categorized into:

- Core Utilities & Assistants: e.g. calculators, date/time fetchers, or weather checkers.
- Knowledge Tools: e.g. search agents, retrieval systems, or Q&A over documents.
- LLM-Based Agents: e.g. chain-of-thought reasoners, planners, or local copilots.
- Interoperability Bridges: e.g. wrappers around third-party APIs or cross-agent protocols.
- Developer Tools: e.g. for testing, visualizing, or debugging MCP workflows.

The Model Context Protocol (MCP) standardizes how such tools are exposed and accessed — allowing systems to discover, describe, and invoke tools in a stateless, modular, and often distributed fashion. In practical terms, MCP provides the glue between an intelligent system and its environment, decoupling the “brains” (the LLM system) from its tools in both time and location.

To realize this, the MCP ecosystem typically involves three architectural roles:

### MCP Server

An MCP server hosts tools. Each tool exposes a defined interface (name, parameters, description), and responds to standard MCP messages (like initialize, tools/list, tools/call). These servers can run locally, on remote machines, or be embedded in external systems. They are modular, stateless, and reusable: anything, anywhere.

### MCP Client

A client establishes the actual connection to one or more servers. It sends requests, handles responses, and often abstracts transport protocols (like stdin/stdout, SSE, or http-stream). Clients are typically “thin”: they act as low-level connectors between tools and the system (host or agent) that wants to use them.

### MCP Host

The host coordinates everything. It’s not a "host" in the traditional networking sense (like a physical machine), but rather the central component in an LLM system — typically the one talking to the LLM itself. It initializes client connections, discovers available tools, and formulates tool calls based on user needs or LLM decisions.

Think of it as the hub that enables the LLM to interact with the right tools at the right time.

Here’s a simplified overview of the architecture we'll build (up to the actual app / user interaction):

```
   ┌──────────────┐       ┌───────────────┐
   │  MathServer  │       │ WeatherServer │
   │  [add, mult] │       │ [get_weather] │
   └──────┬───────┘       └──────┬────────┘
          │                      │
   ┌──────▼──────────────────────▼──────┐
   │            MCP Client              │
   │ (connects to both servers, routes  │
   │  tool calls via common interface)  │
   └──────────────┬─────────────────────┘
                  │
         ┌────────▼─────────┐
         │      MCP Host    │
         │ (talks with LLM) │
         └────────┬─────────┘
                  │
               ┌──▼──┐
               │ App │
               └──┬──┘
                  │
                 User


Architecture of a minimal MCP multi-server setup
```

In the following sections, we'll implement this architecture using the lower-level mcp.server.lowlevel, mcp.client.session, and related classes from the Python MCP SDK. Rather than relying on high-level wrappers like fastmcp, we’ll stay closer to the metal to give you insight into how things actually work.

To keep things simple and focused, we’ll use tools from the Core Utilities & Assistants category — enough to demonstrate real, working multi-server setups without overwhelming complexity.

## Exemplary MCP Servers Using the Python MCP SDK

As mentioned earlier, we’ll use the lower-level MCP SDK classes directly — especially `Server` and `SseServerTransport`. This gives us full control over server behavior and lets us see exactly how a tool is made discoverable and callable via the Model Context Protocol.

> Everything shown below is based on **MCP SDK version 1.6.0**.  
> Install it with e.g.:
>
> ```bash
> pip install mcp==1.6.0
> ```

Additional libraries (like starlette, uvicorn, or openai) are used throughout the examples. These will need to be installed as well, though we won’t call out each one individually.

### 1. Basic Server Setup

To begin, we import and instantiate the server:

```python
from mcp.server.lowlevel import Server
server = Server("MathServer")
```

This registers a new server with the name `MathServer`.

### 2. Define Your Tool(s)

We define each tool using plain Python functions — no decorators needed. What matters is:

- Type hints for all parameters and return values.
- A clean docstring describing its behavior (this is helpful for downstream tool summaries).

```python
def add(a: int, b: int) -> int:
    """Adds two integers.

    Args:
         a: The first integer.
         b: The second integer.
    """
    return a + b
```

### 3. Tool Discovery

MCP needs a way to “see” which tools a server supports. This is done with a `@server.list_tools()` function that returns a list of `Tool` objects — each specifying name, description, input schema, etc.

```python
import mcp.types as types

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
            inputSchema={ ... },
        ),
    ]
```

### 4. Tool Invocation

Now we need to define *how* the server will actually execute tool calls. This is done with the `@server.call_tool()` decorator:

```python
from mcp.types import TextContent

@server.call_tool()
async def handle_call(func_name: str, args: dict):
    if func_name not in ["add", "multiply"]:
        raise ValueError("Unknown tool")

    func = globals().get(func_name)
    if callable(func):
        result = func(**args)
        return [TextContent(type="text", text=str(result))]
    else:
        return [TextContent(type="text", text=f"Function {func_name} not defined.")]
```

> Note: We wrap the result in a list of `TextContent`, which is the expected format for returning tool output. Even if you only return a string, it must be inside a `TextContent` and placed in a list — otherwise, the call will fail.

### 5. Serving Over SSE with Starlette

To enable real-time communication, we expose our server over **Server-Sent Events (SSE)** using an ASGI-compatible app — here, we use [Starlette](https://www.starlette.io/) because it's lightweight and already integrated with MCP SDK classes.

> ⚠️ **Note:** SSE transport is now deprecated in favor of [Streamable HTTP](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206), but still supported in MCP v1.6.0.

#### Define the transport:

```python
from mcp.server.sse import SseServerTransport

sse_transport = SseServerTransport("/messages/")
```

#### Route SSE requests to your server:

```python
async def sse_handler(request):
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
```

#### Define Starlette routes:

```python
from starlette.applications import Starlette
from starlette.routing import Route, Mount

routes = [
    Route("/sse", endpoint=sse_handler),
    Mount("/messages/", app=sse_transport.handle_post_message),
]

starlette_app = Starlette(routes=routes)
```

### 6. Start the Server

Finally, launch the ASGI app with `uvicorn`, binding it to your chosen port (e.g. 58000):

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=58000)
```

### 7. Run and Extend

You can now save the script as `math_server.py` and start it with:

```bash
python math_server.py
```

We also define a second server — `weather_server.py` — listening on a different port (e.g., 58001) and offering a fake weather tool. That way, we’ll be able to run multiple servers in parallel on the same machine (or across machines, if desired).

The full code — including both servers, client setup, and host logic — is available in our accompanying GitHub repository: [mcp-multi-server-demo](https://github.com/SeaSparks-GmbH/mcp-multi-server-demo).

## Exemplary Client

Now that the servers are running, we need to establish the "thin" client that allows the host to connect to the servers and their tools in a standardized way.

To handle the connection, sessions and communication with the servers, we use and start with setting up a MultiServerClient class, which will hold all methods needed to  
- initialize the connection to the servers  
- request a list of all the tools that are served by the servers  
- place calls to the tools from the host and communicate the results of the calls back to the host  
- close the sessions gracefully

The `__init__` just sets the stage for the connection to the servers (sse-endpoints) and the exit.

```python
from contextlib import AsyncExitStack

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
```

Next, we define compact methods within the class to connect to all servers (initialization via SSE) and to close them when the connection is no longer needed. We use two lower-level MCP SDK elements for this: `sse_client` and `ClientSession`:

```python
from mcp.client.sse import sse_client
from mcp import ClientSession

async def connect_all(self):
    for name, sse_url in self.endpoints.items():
        read, write = await self._exit_stack.enter_async_context(sse_client(sse_url))
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.sessions[name] = session
        
async def disconnect_all(self):
    await self._exit_stack.aclose()
```

> **A Note on `ClientSession` and `sse_client`**
>
> We're working directly with `ClientSession` and `sse_client` here — rather than using a high-level abstraction like `fastmcp` — for a reason.
>
> - `sse_client()` opens a Server-Sent Events connection and returns a read/write stream pair.
> - `ClientSession` wraps those streams into a structured MCP session, offering methods like `initialize()`, `list_tools()`, and `call_tool()`.
>
> Libraries like `fastmcp` abstract this away, but doing it manually helps demystify how MCP “talks” — and shows just how lightweight and modular the protocol actually is.


Finally, we need a method that produces a list of all tools from all server connections and another method to route a call to a given tool:

```python
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
```

That's already it for the client. We now move on to the third and final building block, the host.

## Exemplary Host

To complete our MCP architecture, we need a way to query an LLM and let it decide which tool to invoke. In this example, we'll use OpenAI's API, but you can easily swap it for any other LLM service or local model server—as long as the model is powerful enough to produce structured output and undeerstand our basic instructions.

```python
from openai import AsyncOpenAI

# Replace this with your real key
openai_key = "your-api-key-here"

# Method to query the LLM
async def query_llm(prompt):
    llm_client = AsyncOpenAI(api_key=openai_key)
    return await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
```

To keep the host compact and focused, we define a single function that accepts a user query and handles the rest. The function:

- Initializes connections to all servers via our MCP client
- Fetches the available tools from all registered servers
- Constructs a prompt for the LLM including the tools and the user query
- Sends the prompt to the LLM
- Parses the LLM's structured JSON response to determine which tool to call
- Calls the selected tool and returns the result
- Gracefully disconnects from all servers

```python
import json

async def run_host_query(user_input: str):
    endpoints = {
        "math_server": "http://127.0.0.1:58000/sse",
        "weather_server": "http://127.0.0.1:58001/sse"
    }

    mcp_client = MultiServerClient(endpoints)
    print("Connect to servers...")
    await mcp_client.connect_all()

    tools_by_server = await mcp_client.list_all_tools()
    tool_summary = []
    for server, tools in tools_by_server.items():
        for tool in tools:
            tool_summary.append(
                f"server: {server}, tool: {tool.name}, description: {tool.description} input schema: {tool.inputSchema}"
            )
    print("Tool summaries:", tool_summary)

    # Prompt to help LLM choose a tool
    prompt = f"""
You are a tool routing assistant. Choose the best tool for the user query. Available tools:

{chr(10).join(tool_summary)}

Given the user query: \"{user_input}\"
Respond only with a JSON object like:
{{"server": "math_server", "tool": "add", "args": {{"a": 3, "b": 5}}}}
"""

    response = await query_llm(prompt)
    raw_content = response.choices[0].message.content

    # Clean markdown formatting if included
    cleaned_content = raw_content.strip().strip("```json").strip("```").strip()
    parsed = json.loads(cleaned_content)
    print("LLM chose:", parsed)

    result = await mcp_client.call(parsed["server"], parsed["tool"], parsed["args"])
    print("Tool result:", result)

    await mcp_client.disconnect_all()
```

This host logic works well in a Jupyter notebook (useful if you want to adjust settings or try out things) or Python script. It’s intended to be a simple but clear, sequential example of how an LLM-powered host can dynamically interact with modular MCP services based on a single user query.

## Results and Wrap-Up

Once everything is up and running, you can call something like:

```python
await run_host_query("How many hours do 5 days have?")
```

And from the print statements, you'll get output like this:

```
Connect to servers...
Tool summaries: [
  "server: math_server, tool: add, description: Adds two integers input schema: {'type': 'object', 'required': ['a', 'b'], 'properties': {'a': {'type': 'integer', 'description': 'First number'}, 'b': {'type': 'integer', 'description': 'Second number'}}}", 
  "server: math_server, tool: multiply, description: Multiplies two integers input schema: {'type': 'object', 'required': ['a', 'b'], 'properties': {'a': {'type': 'integer', 'description': 'First number'}, 'b': {'type': 'integer', 'description': 'Second number'}}}", 
  "server: weather_server, tool: get_weather, description: Returns fake weather input schema: {'type': 'object', 'required': ['city'], 'properties': {'city': {'type': 'string', 'description': 'City to get the weather for'}}}"
]
LLM chose: {'server': 'math_server', 'tool': 'multiply', 'args': {'a': 5, 'b': 24}}
Tool result: meta=None content=[TextContent(type='text', text='120', annotations=None)] isError=False
```

Here's what we see:
- The servers respond to the `list_tools` request via the client, exactly as expected.
- The LLM, using that tool metadata, selects the appropriate tool for the task.
- The tool is called based on the LLM’s structured output — and gives the correct answer: 5 × 24 = **120** hours.

## What’s Next?

Our setup above is intentionally minimal. It’s meant to *illustrate* the inner workings of MCP using multiple servers and low-level SDK constructs. That clarity comes at the cost of abstraction — which is where higher-level libraries like `FastMCP` typically shine.

Still, there’s plenty of room to build on what we’ve done:  
- Automatically register tools via introspection, as hinted in the server section — to reduce redundancy and streamline tool definition.
- Replace SSE with Streamable HTTP transport, the modern alternative now supported in MCP. (We may explore this in a follow-up article.)
- Swap in your own LLM, whether it's a local model or a hosted one — or even build a fully self-hosted chain-of-reasoning setup.
- Integrate libraries like LangChain or LangGraph to formalize and extend the system — using structured format instructions derived from tool metadata, and building more robust, composable prompting workflows.

We’d love to hear your thoughts — especially if you’re experimenting with MCP in more complex or production-grade setups. Just [Email us](mailto:kontakt@seasparks.de).

Thanks for following along — and stay curious. Our AI journey is just getting started.

