import asyncio
from graph import Rag_Graph
from langchain_core.messages import HumanMessage

async def main():
    app = Rag_Graph().get_app
    config = {"configurable": {"thread_id": "conversation_123"}}

    # Turn 1
    print("=== Turn 1 ===")
    response = await app.ainvoke(
        {"messages": [HumanMessage(content="What is LangGraph?")]},
        config=config
    )
    print(response["messages"][-1].content)

    # Turn 2
    print("\n=== Turn 2 ===")
    response = await app.ainvoke(
        {"messages": [HumanMessage(content="What did I ask just now")]},
        config=config
    )
    print(response["messages"][-1].content)

    # Turn 3
    print("\n=== Turn 3 ===")
    response = await app.ainvoke(
        {"messages": [HumanMessage(content="What did I ask just now")]},
        config={"configurable": {"thread_id": "conversation_456"}}
    )
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
