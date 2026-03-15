import asyncio
from graph import Rag_Graph
from langchain_core.messages import HumanMessage


async def stream_query(app, query, config):
    async for event in app.astream_events(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        version="v2"
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node") == "generate"
        ):
            chunk = event["data"]["chunk"]
            content = chunk.content if isinstance(chunk.content, str) else (chunk.content[0].get("text", "") if chunk.content else "")
            if content:
                print(content, end="", flush=True)
    print()


async def main():
    app = Rag_Graph().get_app
    config = {"configurable": {"thread_id": "conversation_123"}}

    # Turn 1
    print("=== Turn 1 ===")
    await stream_query(app, "What is LangGraph?", config)

    # Turn 2
    print("\n=== Turn 2 ===")
    await stream_query(app, "What did I ask just now", config)

    # Turn 3
    print("\n=== Turn 3 ===")
    await stream_query(app, "What did I ask just now", {"configurable": {"thread_id": "conversation_456"}})


if __name__ == "__main__":
    asyncio.run(main())
