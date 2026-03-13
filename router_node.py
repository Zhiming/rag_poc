from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from model import AgentState


class RouteDecision(BaseModel):
    needs_retrieval: bool


class Router_Node():
    def __init__(self, llm):
        self.__llm = llm.with_structured_output(RouteDecision)

    async def route(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        system_prompt = SystemMessage("""You are a routing assistant.
Decide if the user's question requires searching external documents, or if it can be answered from the conversation history alone.
Set needs_retrieval to true if external documents are needed, false if the conversation history is sufficient.""")

        decision = await self.__llm.ainvoke([system_prompt, last_message])

        return {"needs_retrieval": decision.needs_retrieval}
