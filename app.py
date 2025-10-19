import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from lib.index_builder import index_loader
from lib.graph_builder import work_flow
from lib.retrieval_grader import grade_checker
from lib.rewriter import rewrite_question, initialize_rewriter
from lib.generate import generation
from lib.router import router_calling
from lib.hallucination_grader import hallucination_checker
from lib.answer_grader import Ans_checker

st.set_page_config(
    page_title="Research Paper Chatbot - Cross-Document Analysis",
    page_icon="📚",
    layout="wide"
)

def initialize_app(force_rebuild=False):
    """Initialize the LangGraph app with all components"""
    load_dotenv()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    initialize_rewriter(llm)
    
    retriever = index_loader(force_rebuild=force_rebuild)
    
    from lib.generate import set_global_retriever
    set_global_retriever(retriever)
    
    app = work_flow(
        retriever=retriever,
        generation_func=generation,
        grade_checker_func=grade_checker,
        rewrite_question_func=rewrite_question,
        router_calling_func=router_calling,
        hallucination_checker_func=hallucination_checker,
        ans_checker_func=Ans_checker
    )
    
    return app, retriever

def process_question(app, prompt):
    """Process a question and return the result"""
    try:
        inputs = {"question": prompt}
        
        # Create placeholder for processing steps
        steps_placeholder = st.empty()
        steps_text = ""
        
        with st.expander("Processing Steps", expanded=False):
            steps_container = st.container()
            
            final_value = None
            for output in app.stream(inputs, config={"recursion_limit": 50}):
                for key, value in output.items():
                    steps_text += f"{key} -> "
                    steps_container.markdown(f"```\n{steps_text}\n```")
                    final_value = value
            
            # Show paper diversity info
            if final_value and final_value.get("documents"):
                docs = final_value["documents"]
                unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in docs)
                st.info(f"Retrieved information from {len(unique_papers)} papers")
                
                if len(unique_papers) > 1:
                    with st.expander("Papers Referenced"):
                        for paper in sorted(unique_papers):
                            st.write(f"• {paper}")
        
        if final_value and "generation" in final_value:
            return final_value["generation"], None
        else:
            return None, "No answer was generated."
            
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    st.title("Research Paper Chatbot")
    st.markdown("**Enhanced with Cross-Document Analysis** - Ask questions across all 14 research papers!")
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'force_rebuild' not in st.session_state:
        st.session_state.force_rebuild = False
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = None
    
    # Load app
    with st.spinner("Loading chatbot with enhanced hybrid retrieval..."):
        app, retriever = initialize_app(force_rebuild=st.session_state.force_rebuild)
        st.session_state.force_rebuild = False
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot uses **Enhanced Hybrid Retrieval** to:
        - Compare information across multiple papers
        - Analyze trends and patterns
        - Provide comprehensive cross-document insights
        - Access all 14 research papers simultaneously
        """)
        
        st.header("Cross-Document Examples")
        
        # Comparison Queries with expander
        with st.expander("Comparison Queries"):
            comparison_queries = [
                "Compare accuracies across all papers",
                "Which paper achieved highest accuracy?",
                "Compare CNN vs traditional methods",
                "Compare drone-based vs mobile detection"
            ]
            
            for query in comparison_queries:
                if st.button(query, key=f"comp_{query[:30]}", use_container_width=True):
                    st.session_state.pending_question = query
                    st.rerun()
        
        # Aggregative Queries with expander
        with st.expander("Aggregative Queries"):
            aggregative_queries = [
                "What are common methods used?",
                "How many papers used drones?",
                "What's the typical accuracy range?",
                "List all deep learning architectures"
            ]
            
            for query in aggregative_queries:
                if st.button(query, key=f"agg_{query[:30]}", use_container_width=True):
                    st.session_state.pending_question = query
                    st.rerun()
        
        # Listing Queries with expander
        with st.expander("Listing Queries"):
            listing_queries = [
                "List all paper titles",
                "Which papers used thermal imaging?",
                "Papers with accuracy above 90%",
                "All datasets mentioned"
            ]
            
            for query in listing_queries:
                if st.button(query, key=f"list_{query[:30]}", use_container_width=True):
                    st.session_state.pending_question = query
                    st.rerun()
        
        # Technical Comparisons with expander
        with st.expander("Technical Comparisons"):
            technical_queries = [
                "Compare preprocessing techniques",
                "Dataset size comparison",
                "Common data augmentation methods",
                "Performance vs computational cost"
            ]
            
            for query in technical_queries:
                if st.button(query, key=f"tech_{query[:30]}", use_container_width=True):
                    st.session_state.pending_question = query
                    st.rerun()
        
        st.header("Chat History")
        if st.session_state.history:
            for i, msg in enumerate(st.session_state.history[-5:]):
                st.text(f"{i+1}. {msg[:50]}...")
        else:
            st.text("No chat history yet.")
    
    # Process pending question from sidebar
    if st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None
        
        # Add to conversation
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.session_state.history.append(prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing across all papers..."):
                answer, error = process_question(app, prompt)
                
                if answer:
                    st.markdown(answer)
                    st.session_state.conversation.append({
                        "role": "assistant", 
                        "content": answer
                    })
                else:
                    error_msg = error or "Sorry, I couldn't generate an answer."
                    st.error(error_msg)
                    st.session_state.conversation.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Display existing conversation
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question (supports cross-document analysis)..."):
        # Add user message
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.session_state.history.append(prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing across all papers..."):
                answer, error = process_question(app, prompt)
                
                if answer:
                    st.markdown(answer)
                    st.session_state.conversation.append({
                        "role": "assistant", 
                        "content": answer
                    })
                else:
                    error_msg = error or "Sorry, I couldn't generate an answer."
                    st.error(error_msg)
                    st.session_state.conversation.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
        
        st.rerun()
    
    # Bottom controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Clear Conversation"):
            st.session_state.conversation = []
            st.session_state.history = []
            st.rerun()
    
    with col2:
        if st.button("Rebuild Database"):
            with st.spinner("Rebuilding database..."):
                try:
                    st.session_state.force_rebuild = True
                    st.success("Database rebuild initiated")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")
    
    with col3:
        if st.button("Show Statistics"):
            with st.spinner("Gathering statistics..."):
                try:
                    titles = retriever.get_all_paper_titles()
                    st.info(f"""
                    **Database Statistics:**
                    - Total Papers: {len(titles)}
                    - Retrieval Method: Enhanced Hybrid (Dense + Sparse)
                    - Cross-Document Analysis: Enabled
                    """)
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()