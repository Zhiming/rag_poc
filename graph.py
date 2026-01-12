from model import AgentState
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from response_node import Response_Node
from retriever_node import Retreiever_Node
from embedding_model import Context_Retriever
from chat_model import Chat_Model


class Rag_Graph():
    def __init__(self):
        workflow = StateGraph(AgentState)

        # Set up retriever using the existing Context_Retriever
        context_retriever = Context_Retriever()

        retriever_node = Retreiever_Node(context_retriever.retriever)
        chat_model = Chat_Model()
        response_node = Response_Node(chat_model.chat_llm)
        
        # Step 2: Add nodes by passing METHOD REFERENCES (not calling them)
        workflow.add_node("retrieve", retriever_node.retrieve_context)
        workflow.add_node("generate", response_node.generate_response)
        
        # Step 3: Set up the flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        memory = MemorySaver()
        self.__app = workflow.compile(checkpointer=memory)

    @property
    def get_app(self):
        return self.__app
