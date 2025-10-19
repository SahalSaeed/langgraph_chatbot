"""Query Rewriter - Optimizes questions for better retrieval"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

question_rewriter = None

def create_question_rewriter(llm=None):
    """Create a question rewriter with optional LLM"""
    from langchain_openai import ChatOpenAI
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if llm is None:
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    system = """You are a question re-writer that converts an input question to a better version that is optimized 
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )

    return re_write_prompt | llm | StrOutputParser()

def initialize_rewriter(llm):
    """Initialize the question rewriter with a specific LLM"""
    global question_rewriter
    question_rewriter = create_question_rewriter(llm)

def rewrite_question(question):
    """Rewrite a question to be more optimized for retrieval"""
    global question_rewriter
    
    if question_rewriter is None:
        question_rewriter = create_question_rewriter()
    
    return question_rewriter.invoke({"question": question})