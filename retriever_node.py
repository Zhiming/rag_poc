from model import AgentState


class Retreiever_Node():
    def __init__(self, vector_retriever):
        self.__vector_retriever = vector_retriever


    def retrieve_context(self, state: AgentState):
        last_message = state["messages"][-1]
        
        docs = self.__vector_retriever.invoke(last_message.content)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        return {"context": context}