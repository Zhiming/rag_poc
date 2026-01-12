from langchain_core.messages import HumanMessage
from model import AgentState


class User_Input_Node():
    def _init__(self, user_input):
        pass
    
    def get_user_input(self, state: AgentState, user_input: str):
        return {"message", [HumanMessage(user_input)]}