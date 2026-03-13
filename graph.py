from model import AgentState
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from response_node import Response_Node
from retriever_node import Retreiever_Node
from embedding_model import Context_Retriever
from chat_model import Chat_Model
from router_node import Router_Node


class Rag_Graph():
    def __init__(self):
        workflow = StateGraph(AgentState)

        context_retriever = Context_Retriever()
        chat_model = Chat_Model()

        router_node = Router_Node(chat_model.chat_llm)
        retriever_node = Retreiever_Node(context_retriever.retriever)
        response_node = Response_Node(chat_model.chat_llm)

        workflow.add_node("router", router_node.route)
        workflow.add_node("retrieve", retriever_node.retrieve_context)
        workflow.add_node("generate", response_node.generate_response)

        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            lambda state: "retrieve" if state["needs_retrieval"] else "generate"
        )
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        memory = MemorySaver()
        self.__app = workflow.compile(checkpointer=memory)

    @property
    def get_app(self):
        return self.__app
