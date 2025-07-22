# nodes/route_mode_node.py
from typing import Dict

def route_mode_node(state: Dict) -> Dict:
    """
    Decides whether to route to VQA or RAG based on file type
    and returns a dict including the routing decision.
    """
    if state.get("is_image", False):
        print("Route mode node returning: vqa_tool")
        return {"__next__": "vqa_tool", **state} # Return a dict with __next__ key
    else:
        print("Route mode node returning: rag_tool")
        return {"__next__": "rag_tool", **state} # Return a dict with __next__ key