"""Router - Routes queries to appropriate datasource"""
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import dotenv

dotenv.load_dotenv()

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore"] = Field(
        ...,
        description="Given a user question to route it to a vectorstore.",
    )

def router_calling(query):
    """Route the query to the appropriate datasource"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system = """You are an expert at routing a user question to a vectorstore.
    The vectorstore contains documents related to potholes, road cracks, road conditions, accuracies, methods and algorithms.
    Use the vectorstore for questions on these topics."""
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router
    result = question_router.invoke({"question": query})
    
    return result