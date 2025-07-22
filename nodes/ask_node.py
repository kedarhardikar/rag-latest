# nodes/ask_node.py
from typing import Dict
import logging

# Configure logging at the module level (or centrally in your app)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def ask_node(state: Dict) -> Dict:
    """
    A simple node that acknowledges the user's input (question)
    and passes the state forward.
    """
    user_input = state.get("input", "")
    file_path = state.get("file_path", "")

    logging.info(f"Ask Node: Received input: '{user_input}' for file: '{file_path}'")

    # In a more complex scenario, you might do initial parsing,
    # validation, or simple response generation here.
    # For now, it just passes the state through.
    return {**state}