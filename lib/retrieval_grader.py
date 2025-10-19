"""Retrieval Grader - Checks document relevance to question"""
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def create_retrieval_grader():
    """Create and return a retrieval grader function"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    
    For questions about research paper titles, be more lenient - if the document contains research content,
    technical methods, or accuracy results, consider it relevant."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    return grade_prompt | structured_llm_grader

def grade_checker(documents, question):
    """
    Grade documents for relevance to the question.
    
    Args:
        documents: List of documents to grade
        question: The user's question
    
    Returns:
        filtered_docs: List of relevant documents
        question: The original question
    """
    retrieval_grader = create_retrieval_grader()
    
    if not documents:
        print("No documents to grade.")
        return [], question
    
    print(f"Grading {len(documents)} documents for relevance")
    
    is_title_question = any(keyword in question.lower() for keyword in [
        "title", "research paper", "which paper", "what paper", "name of the paper"
    ])
    
    filtered_docs = []
    for doc in documents:
        if is_title_question:
            content = doc.page_content.lower()
            if any(indicator in content for indicator in [
                "research", "paper", "study", "method", "accuracy", "detection",
                "model", "technique", "algorithm", "deep learning", "pothole", "crack"
            ]):
                print("---GRADE: DOCUMENT RELEVANT (title question lenient grading)---")
                filtered_docs.append(doc)
                continue
        
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    
    print(f"Filtered to {len(filtered_docs)} relevant documents")
    return filtered_docs, question