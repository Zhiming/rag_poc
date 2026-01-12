import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage


class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    context: str