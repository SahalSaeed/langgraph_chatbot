from ..lib.index_builder import index_loader
from ..lib.graph_builder import work_flow
from pprint import pprint
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ..lib.retrieval_grader import grade_checker
from ..lib.rewriter import rewrite_question, initialize_rewriter
from ..lib.generate import generation
from ..lib.router import router_calling
from ..lib.answer_grader import Ans_checker
from ..lib.hallucination_grader import hallucination_checker


def save_graph_image(app, output_path="workflow_graph.png"):
    """Save the workflow graph as a PNG image"""
    try:
        # Get the graph PNG data
        graph_png = app.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(graph_png)
        
        print(f"Workflow graph saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving graph image: {e}")
        return False

def main():
    load_dotenv()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    initialize_rewriter(llm)
    
    try:
        retriever = index_loader()
        print("Retriever initialized successfully with hybrid retrieval")
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return
    
    app = work_flow(
        retriever=retriever,
        generation_func=generation,
        grade_checker_func=grade_checker, 
        rewrite_question_func=rewrite_question,
        router_calling_func=router_calling,
        hallucination_checker_func=hallucination_checker,
        ans_checker_func=Ans_checker
    )
    
    # Save the workflow graph as PNG
    save_graph_image(app, "workflow_graph.png")
    
    conversation_history = []
    
    while True:    
        query = input("\nAsk your question (or type 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        inputs = {
            "question": query,
            "conversation_history": conversation_history.copy(),
            "transform_count": 0,
            "hallucination_retry_count": 0,
            "documents": [],
            "generation": ""
        }
        
        try:
            final_state = None
            for output in app.stream(inputs, config={"recursion_limit": 50}):
                for key, value in output.items():
                    pprint(f"Node '{key}':")
                pprint("\n---\n")
                final_state = value
            
            if final_state and "conversation_history" in final_state:
                conversation_history = final_state["conversation_history"]
            
            if final_state and "generation" in final_state:
                print("\n=== ANSWER ===")
                print(final_state["generation"])
                print("==============")
            else:
                print("\nNo answer generated.")
                
        except Exception as e:
            print(f"Error during processing: {e}")
            fallback_answer = "I encountered an error. Let me try to answer based on our conversation."
            if conversation_history:
                print(f"\n=== ANSWER ===\n{fallback_answer}")
                print("I recall from our previous discussion that we talked about research papers on road detection methods.")
                print("==============")
            else:
                print(f"\n=== ANSWER ===\n{fallback_answer} Please try rephrasing your question.")
                print("==============")

if __name__ == "__main__":
    main()