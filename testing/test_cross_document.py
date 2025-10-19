"""
Test Script for Cross-Document Analysis Features
Run this to verify your enhanced system is working correctly
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ..lib.index_builder import index_loader
from ..lib.graph_builder import work_flow
from ..lib.retrieval_grader import grade_checker
from ..lib.rewriter import rewrite_question, initialize_rewriter
from ..lib.generate import generation
from ..lib.router import router_calling
from ..lib.answer_grader import Ans_checker
from ..lib.hallucination_grader import hallucination_checker

def test_cross_document_features():
    """Test all cross-document capabilities"""
    
    print("=" * 70)
    print("CROSS-DOCUMENT ANALYSIS TESTING")
    print("=" * 70)
    
    # Initialize system
    print("\n[1/6] Initializing system...")
    load_dotenv()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    initialize_rewriter(llm)
    retriever = index_loader()
    
    from ..lib.generate import set_global_retriever
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
    
    print("✅ System initialized successfully")
    
    # Test 1: Check retriever features
    print("\n[2/6] Testing retriever features...")
    try:
        all_titles = retriever.get_all_paper_titles()
        print(f"✅ Found {len(all_titles)} papers in index")
        
        # Test get_papers_by_topic
        drone_papers = retriever.get_papers_by_topic("drone")
        print(f"✅ Found {len(drone_papers)} papers about drones")
        
        # Test cross-document retrieval
        query = "accuracy comparison"
        docs = retriever.invoke(query, cross_document=True)
        unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in docs)
        print(f"✅ Cross-document retrieval returned {len(unique_papers)} unique papers")
        
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return False
    
    # Test 2: Comparative query
    print("\n[3/6] Testing comparative query...")
    test_query_1 = "Compare the accuracy achieved across different papers"
    
    try:
        inputs = {"question": test_query_1}
        result = None
        
        for output in app.stream(inputs, config={"recursion_limit": 50}):
            for key, value in output.items():
                result = value
        
        if result and "generation" in result:
            answer = result["generation"]
            print(f"✅ Generated comparative answer ({len(answer)} chars)")
            
            # Check if multiple papers are mentioned
            papers_mentioned = sum(1 for title in all_titles if title.lower() in answer.lower())
            print(f"✅ Answer mentions {papers_mentioned} papers")
            
            if papers_mentioned < 2:
                print("⚠️  Warning: Expected multiple papers in comparative answer")
        else:
            print("❌ No answer generated")
            return False
            
    except Exception as e:
        print(f"❌ Comparative query test failed: {e}")
        return False
    
    # Test 3: Aggregative query
    print("\n[4/6] Testing aggregative query...")
    test_query_2 = "What are the most common methods used across all papers?"
    
    try:
        inputs = {"question": test_query_2}
        result = None
        
        for output in app.stream(inputs, config={"recursion_limit": 50}):
            for key, value in output.items():
                result = value
        
        if result and "generation" in result:
            answer = result["generation"]
            print(f"✅ Generated aggregative answer ({len(answer)} chars)")
            
            # Check for method keywords
            method_keywords = ['cnn', 'yolo', 'deep learning', 'machine learning']
            methods_found = sum(1 for keyword in method_keywords if keyword in answer.lower())
            print(f"✅ Answer mentions {methods_found} common methods")
        else:
            print("❌ No answer generated")
            return False
            
    except Exception as e:
        print(f"❌ Aggregative query test failed: {e}")
        return False
    
    # Test 4: Listing query
    print("\n[5/6] Testing listing query...")
    test_query_3 = "List papers that used drone-based detection"
    
    try:
        inputs = {"question": test_query_3}
        result = None
        
        for output in app.stream(inputs, config={"recursion_limit": 50}):
            for key, value in output.items():
                result = value
        
        if result and "generation" in result:
            answer = result["generation"]
            print(f"✅ Generated listing answer ({len(answer)} chars)")
            
            # Check if papers are listed
            if any(char.isdigit() and ('. ' in answer or ') ' in answer) for char in answer):
                print("✅ Answer appears to contain a formatted list")
            else:
                print("⚠️  Warning: Answer may not be properly formatted as list")
        else:
            print("❌ No answer generated")
            return False
            
    except Exception as e:
        print(f"❌ Listing query test failed: {e}")
        return False
    
    # Test 5: Document diversity check
    print("\n[6/6] Testing document diversity...")
    test_query_4 = "Compare CNN-based and traditional methods"
    
    try:
        inputs = {"question": test_query_4}
        result = None
        
        for output in app.stream(inputs, config={"recursion_limit": 50}):
            for key, value in output.items():
                result = value
        
        if result and "documents" in result:
            docs = result["documents"]
            unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in docs)
            
            print(f"✅ Retrieved {len(docs)} chunks from {len(unique_papers)} papers")
            
            if len(unique_papers) >= 3:
                print("✅ Good paper diversity (3+ papers)")
            elif len(unique_papers) >= 2:
                print("⚠️  Moderate paper diversity (2 papers)")
            else:
                print("⚠️  Warning: Low paper diversity (1 paper)")
        else:
            print("❌ No documents in result")
            return False
            
    except Exception as e:
        print(f"❌ Document diversity test failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✅ All tests passed!")
    print("\nYour enhanced cross-document system is working correctly.")
    print("\nYou can now:")
    print("  • Compare information across multiple papers")
    print("  • Get aggregated statistics and trends")
    print("  • List and filter papers by criteria")
    print("  • Perform comprehensive literature analysis")
    print("\nRun 'streamlit run app.py' to use the enhanced chatbot!")
    print("=" * 70)
    
    return True

def interactive_test():
    """Interactive testing mode"""
    print("\n" + "=" * 70)
    print("INTERACTIVE CROSS-DOCUMENT TESTING")
    print("=" * 70)
    
    # Initialize system
    print("\nInitializing system...")
    load_dotenv()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    initialize_rewriter(llm)
    retriever = index_loader()
    
    from ..lib.generate import set_global_retriever
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
    
    print("✅ System ready!\n")
    
    # Show example queries
    print("Example cross-document queries:")
    print("  1. Compare accuracies across all papers")
    print("  2. What are the most common methods?")
    print("  3. List papers using deep learning")
    print("  4. Which paper has the highest accuracy?")
    print("  5. Compare drone-based vs mobile detection")
    print("\nType 'exit' to quit\n")
    
    while True:
        query = input("Your question: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue
        
        print("\nProcessing...")
        
        try:
            inputs = {"question": query}
            result = None
            
            for output in app.stream(inputs, config={"recursion_limit": 50}):
                for key, value in output.items():
                    result = value
            
            if result and "generation" in result:
                print("\n" + "=" * 70)
                print("ANSWER:")
                print("=" * 70)
                print(result["generation"])
                print("=" * 70)
                
                # Show document diversity
                if "documents" in result:
                    docs = result["documents"]
                    unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in docs)
                    print(f"\n[Retrieved from {len(unique_papers)} papers: {len(docs)} chunks total]")
                
            else:
                print("\n❌ No answer generated")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        success = test_cross_document_features()
        sys.exit(0 if success else 1)