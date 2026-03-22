from functools import partial

from langgraph.graph import END, StateGraph

from graph.config.chat_model import Chat_Model
from graph.constants import FIELD_VALIDATION_ERRORS, NODE_INVOKE_LLM, NODE_VALIDATE_SCHEMA
from graph.evaluation_note_extraction.nodes import invoke_llm, validate_schema
from graph.evaluation_note_extraction.state import EvaluationNoteExtractionState


def _route_after_validation(state: EvaluationNoteExtractionState) -> str:
    if state.get(FIELD_VALIDATION_ERRORS):
        return NODE_INVOKE_LLM
    return END


def build_graph():
    llm = Chat_Model().chat_llm

    builder = StateGraph(EvaluationNoteExtractionState)

    builder.add_node(NODE_INVOKE_LLM, partial(invoke_llm, llm=llm))
    builder.add_node(NODE_VALIDATE_SCHEMA, validate_schema)

    builder.set_entry_point(NODE_INVOKE_LLM)
    builder.add_edge(NODE_INVOKE_LLM, NODE_VALIDATE_SCHEMA)
    builder.add_conditional_edges(NODE_VALIDATE_SCHEMA, _route_after_validation)

    return builder.compile()
