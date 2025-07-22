from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Any
from dotenv import load_dotenv

from vqa_tool import vqa_tool_node
from rag_tool import rag_tool_node
from nodes.upload_node import upload
from nodes.ask_node import ask_node
from nodes.route_mode_node import route_mode_node # Keep this import
# import accelerate

# Load .env variables
load_dotenv()

class GraphState(TypedDict):
    input: str
    file_path: str
    is_image: bool
    documents: Optional[list]
    answer: Optional[str]
    vectordb: Optional[Any]
    __next__: Optional[str] # ADD THIS LINE to GraphState

    last_processed_file_path: Optional[str]
    last_processed_file_hash: Optional[str]
    active_collection_name: Optional[str]

def build_graph():
    builder = StateGraph(GraphState)

    # Add all nodes
    builder.add_node("upload", upload)
    builder.add_node("ask_question", ask_node)
    builder.add_node("route_mode", route_mode_node) # This is now a regular state-updating node that also provides routing info
    builder.add_node("rag_tool", rag_tool_node)
    builder.add_node("vqa_tool", vqa_tool_node)

    # Define flow
    builder.set_entry_point("upload")
    builder.add_edge("upload", "ask_question")
    # Instead of conditional_edges, we directly connect to the router node.
    builder.add_edge("ask_question", "route_mode")

    # Now, use `add_conditional_edges` based on the __next__ key in the state
    builder.add_conditional_edges(
        "route_mode", # The source node that outputs the __next__ key
        lambda state: state["__next__"], # The callable to determine the next step
        {
            "rag_tool": "rag_tool",
            "vqa_tool": "vqa_tool",
            END: END # If for some reason __next__ becomes END, handle it.
        }
    )

    # End edges (these remain the same)
    builder.add_edge("rag_tool", END)
    builder.add_edge("vqa_tool", END)

    return builder.compile()