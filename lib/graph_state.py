"""Graph State - Defines the state structure for the workflow"""
from typing import List, Optional
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """State definition for the RAG workflow graph"""
    question: str
    generation: str
    documents: List[str]
    conversation_history: List[dict]
    transform_count: int
    hallucination_retry_count: int
    question_type: Optional[str]