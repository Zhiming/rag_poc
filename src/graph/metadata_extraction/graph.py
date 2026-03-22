from functools import partial

from langgraph.graph import END, StateGraph

from graph.config.chat_model import Chat_Model
from graph.constants import NODE_INVOKE_LLM
from graph.metadata_extraction.nodes import invoke_llm
from graph.metadata_extraction.state import MetadataExtractionState


def build_graph():
    llm = Chat_Model().chat_llm

    builder = StateGraph(MetadataExtractionState)

    builder.add_node(NODE_INVOKE_LLM, partial(invoke_llm, llm=llm))

    builder.set_entry_point(NODE_INVOKE_LLM)
    builder.add_edge(NODE_INVOKE_LLM, END)

    return builder.compile()


graph = build_graph()
