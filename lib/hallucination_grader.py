"""Hallucination Grader - Checks if generation is grounded in facts"""
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

def hallucination_checker(docs, generation_text):
    """Check if the generation is grounded in the retrieved documents"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    is_title_response = any(indicator in generation_text.lower() for indicator in [
        "research paper", "paper titled", "title:", "detection of", "deep learning method",
        "the paper is", "titled"
    ])
    
    if is_title_response:
        system = """You are a grader assessing whether an LLM generation about research paper titles is reasonable given the retrieved facts. 
        
        For title questions, be more lenient. If the generation mentions a plausible research paper title that could reasonably come from the documents, 
        or if the documents contain research content that matches the described methods/accuracy, grade it as 'yes'.
        
        Only grade 'no' if the title is completely fabricated and has no connection to the document content.
        
        Give a binary score 'yes' or 'no'."""
    else:
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
        Give a binary score 'yes' or 'no'. 
        'Yes' means that the answer is GENERALLY grounded in / supported by the set of facts.
        'No' means the answer contains SIGNIFICANT information not found in the facts.
        
        Be reasonable - if the answer captures the main ideas from the facts, grade it as 'yes'.
        Only grade 'no' if there are major contradictions or significant made-up information."""

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts:\n\n{documents}\n\nLLM generation:\n{generation}")
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader

    result = hallucination_grader.invoke({
        "documents": docs,
        "generation": generation_text
    })

    return result