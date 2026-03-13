from langchain_core.messages import SystemMessage
from model import AgentState


class Router_Node():
    def __init__(self, llm):
        self.__llm = llm

    async def route(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        system_prompt = SystemMessage("""You are a routing assistant.
Decide if the user's question requires searching external documents, or if it can be answered from the conversation history alone.
Reply with ONLY one word: 'retrieve' or 'skip'.""")

        response = await self.__llm.ainvoke([system_prompt, last_message])
        needs_retrieval = response.content.strip().lower() == "retrieve"

        return {"needs_retrieval": needs_retrieval}
