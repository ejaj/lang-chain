# test_client.py

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # tell the client where your server is
    server = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:

            # connect to the server
            await session.initialize()

            # ── see all available tools ──
            tools = await session.list_tools()
            print("Tools available:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            # ── call say_hello ──
            result = await session.call_tool(
                "say_hello",
                {"name": "Priya"}
            )
            print("\nResult:", result.content[0].text)
            # Output: Hello, Priya! Welcome!

            # ── call add_numbers ──
            result2 = await session.call_tool(
                "add_numbers",
                {"a": 10, "b": 20}
            )
            print("Sum:", result2.content[0].text)
            # Output: 30

asyncio.run(main())