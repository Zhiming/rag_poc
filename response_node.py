from langchain_core.messages import SystemMessage, AIMessage
from model import AgentState


class Response_Node():
    def __init__(self, llm):
        self.__llm = llm

    async def generate_response(self, state: AgentState):
        """Node that generates response using context and history"""
        context = state.get("context", "")
        messages = state["messages"]

        system_prompt = SystemMessage(f"""You are a helpful assistant. Use the following context to answer questions:

            Context: {context}

            Answer based on the context above and conversation history."""
        )

        all_messages = [system_prompt] + messages
        response = await self.__llm.ainvoke(all_messages)

        return {"messages": [AIMessage(content=response.content)]}
