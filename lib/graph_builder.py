"""Workflow Builder - Creates and compiles the LangGraph workflow"""
from langgraph.graph import END, StateGraph, START
from lib.graph_state import GraphState
from functools import partial

def work_flow(retriever, generation_func, grade_checker_func, rewrite_question_func, 
              router_calling_func, hallucination_checker_func, ans_checker_func):
    """
    Build and compile the RAG workflow graph
    
    Args:
        retriever: Hybrid retriever instance
        generation_func: Function to generate answers
        grade_checker_func: Function to grade document relevance
        rewrite_question_func: Function to rewrite queries
        router_calling_func: Function to route queries
        hallucination_checker_func: Function to check hallucinations
        ans_checker_func: Function to check answer quality
    
    Returns:
        Compiled workflow graph
    """
    workflow = StateGraph(GraphState)

    from lib.graph_flow import (
        retrieve, 
        generate, 
        grade_documents, 
        transform_query, 
        grade_generation_v_documents_and_question, 
        decide_to_generate, 
        route_question
    )

    # Create partial functions with dependencies
    retrieve_with_deps = partial(retrieve, retriever=retriever)
    generate_with_deps = partial(generate, generation_func=generation_func)
    grade_documents_with_deps = partial(grade_documents, grade_checker_func=grade_checker_func)
    transform_query_with_deps = partial(transform_query, rewrite_question_func=rewrite_question_func)
    
    def grade_generation_wrapper(state):
        return grade_generation_v_documents_and_question(
            state, 
            hallucination_checker_func=hallucination_checker_func,
            ans_checker_func=ans_checker_func
        )
    
    def route_question_wrapper(state):
        return route_question(state, router_calling_func=router_calling_func)

    # Add all nodes
    workflow.add_node("retrieve", retrieve_with_deps)
    workflow.add_node("grade_documents", grade_documents_with_deps)
    workflow.add_node("generate", generate_with_deps)
    workflow.add_node("transform_query", transform_query_with_deps)

    # Build the workflow edges
    # Start -> Retrieve (Router is unnecessary since we only have vectorstore)
    workflow.add_edge(START, "retrieve")
    
    # Retrieve -> Grade Documents
    workflow.add_edge("retrieve", "grade_documents")
    
    # Grade Documents -> Conditional (generate or transform_query)
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    
    # Transform Query -> Retrieve (loop back)
    workflow.add_edge("transform_query", "retrieve")
    
    # Generate -> Conditional (check quality)
    workflow.add_conditional_edges(
        "generate",
        grade_generation_wrapper,
        {
            "not supported": "transform_query",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow.compile()