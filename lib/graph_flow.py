import logging

logger = logging.getLogger(__name__)

def detect_cross_document_query(question: str) -> bool:
    """Detect if query requires cross-document analysis"""
    cross_doc_indicators = [
        'compare', 'comparison', 'versus', 'vs', 'difference',
        'all papers', 'across papers', 'between papers',
        'common', 'commonly', 'typical', 'generally',
        'which paper', 'best', 'worst', 'highest', 'lowest',
        'list all', 'what are the', 'how many papers',
        'every paper', 'all 15', 'entire collection'
    ]
    
    question_lower = question.lower()
    return any(indicator in question_lower for indicator in cross_doc_indicators)

def retrieve(state, retriever):
    """Retrieve documents using hybrid retrieval with cross-document awareness"""
    print("---RETRIEVE---")
    question = state["question"]
    conversation_history = state.get("conversation_history", [])
    transform_count = state.get("transform_count", 0)
    
    question_type = classify_question_type(question)
    is_cross_document = detect_cross_document_query(question)
    
    print(f"Question type: {question_type}")
    print(f"Cross-document query: {is_cross_document}")
    
    enhanced_question = question
    
    # Enhanced query for title questions
    if question_type == "title":
        enhanced_question = "research paper title " + question
    
    # Add conversation context
    if conversation_history:
        context_snippets = []
        for msg in conversation_history[-4:]:
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"][:100]
                context_snippets.append(content)
        
        if context_snippets:
            context_text = " | ".join(context_snippets)
            enhanced_question = f"{enhanced_question} [Context: {context_text}]"
    
    print(f"Retrieving with query: {enhanced_question}")
    
    try:
        # Use cross-document retrieval if needed
        if is_cross_document:
            print("Using cross-document retrieval strategy")
            documents = retriever.invoke(enhanced_question, cross_document=True)
            
            # For "all papers" queries, ensure we have comprehensive coverage
            if any(phrase in question.lower() for phrase in ['all papers', 'across all', 'every paper', 'all 14', 'compare accuracies', 'compare all']):
                print("Query requires ALL papers - augmenting retrieval")
                
                # Get all paper titles
                all_titles = retriever.get_all_paper_titles()
                
                # Do targeted retrieval for papers we're missing
                existing_titles = set(doc.metadata.get('title', 'Unknown') for doc in documents)
                missing_titles = [t for t in all_titles if t not in existing_titles]
                
                print(f"Initial retrieval covered {len(existing_titles)}/{len(all_titles)} papers")
                print(f"Missing {len(missing_titles)} papers - augmenting now")
                
                # Get multiple chunks from each missing paper to ensure good coverage
                for title in missing_titles:
                    chunks = retriever.get_paper_chunks(title)
                    if chunks:
                        # Add first 2 chunks from each missing paper for better context
                        documents.extend(chunks[:2])
                        print(f"Added 2 chunks from: {title}")
                
                print(f"Final retrieval: {len(documents)} chunks from {len(set(doc.metadata.get('title', 'Unknown') for doc in documents))} papers")
        else:
            documents = retriever.invoke(enhanced_question)
        
        print(f"Retrieved {len(documents)} documents using hybrid retrieval")
        
        # Log paper diversity
        if documents:
            unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in documents)
            print(f"Documents span {len(unique_papers)} different papers")
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        documents = []
    
    return {
        "documents": documents, 
        "question": question,
        "question_type": question_type,
        "conversation_history": conversation_history,
        "transform_count": transform_count,
        "is_cross_document": is_cross_document
    }

def generate(state, generation_func):
    """Generate answer with question type awareness"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    conversation_history = state.get("conversation_history", [])
    question_type = state.get("question_type", "general")
    transform_count = state.get("transform_count", 0)
    is_cross_document = state.get("is_cross_document", False)
    
    if is_cross_document and documents:
        unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in documents)
        print(f"Generating cross-document answer using {len(unique_papers)} papers")
    
    generate_text = generation_func(documents, question, conversation_history, question_type)
    
    if generate_text is None:
        print("ERROR: Generation function returned None! Using fallback.")
        generate_text = "I apologize, but I couldn't generate an answer based on the available information."
    
    updated_history = conversation_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": generate_text}
    ]
    
    return {
        "documents": documents, 
        "question": question, 
        "generation": generate_text,
        "conversation_history": updated_history,
        "transform_count": transform_count,
        "question_type": question_type,
        "is_cross_document": is_cross_document
    }

def grade_documents(state, grade_checker_func):
    """Determines whether the retrieved documents are relevant to the question"""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    conversation_history = state.get("conversation_history", [])
    transform_count = state.get("transform_count", 0)
    question_type = state.get("question_type", "general")
    is_cross_document = state.get("is_cross_document", False)
    
    # More lenient grading for cross-document queries
    if is_cross_document and documents:
        print("Cross-document query - using lenient grading to preserve paper diversity")
        filtered_docs = documents
        print(f"Keeping {len(filtered_docs)} documents for cross-document analysis")
        
        return {
            "documents": filtered_docs, 
            "question": question,
            "conversation_history": conversation_history,
            "transform_count": transform_count,
            "question_type": question_type,
            "is_cross_document": is_cross_document
        }
    
    # Standard grading for single-paper queries
    if question_type == "title" and documents:
        print("Title question detected - using lenient grading")
        filtered_docs = documents
        print(f"Keeping {len(filtered_docs)} documents for title question")
        
        return {
            "documents": filtered_docs, 
            "question": question,
            "conversation_history": conversation_history,
            "transform_count": transform_count,
            "question_type": question_type,
            "is_cross_document": is_cross_document
        }
    
    filtered_docs, new_question = grade_checker_func(documents, question)
    
    return {
        "documents": filtered_docs, 
        "question": question,
        "conversation_history": conversation_history,
        "transform_count": transform_count,
        "question_type": question_type,
        "is_cross_document": is_cross_document
    }

def transform_query(state, rewrite_question_func):
    """Transform query for better retrieval"""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    conversation_history = state.get("conversation_history", [])
    transform_count = state.get("transform_count", 0) + 1
    question_type = state.get("question_type", "general")
    is_cross_document = state.get("is_cross_document", False)
    
    if transform_count >= 2:
        print("---MAX TRANSFORM COUNT REACHED: GENERATING WITH AVAILABLE DOCUMENTS---")
        return {
            "documents": documents, 
            "question": question, 
            "transform_count": transform_count,
            "conversation_history": conversation_history,
            "question_type": question_type,
            "is_cross_document": is_cross_document
        }
    
    enhanced_question = question
    
    # Add cross-document hint to query transformation
    if is_cross_document:
        enhanced_question = f"{question} (compare across multiple papers)"
    
    # Add conversation context
    if conversation_history:
        recent_context = []
        for msg in conversation_history[-2:]:
            if isinstance(msg, dict) and "content" in msg:
                recent_context.append(msg["content"][:80])
        
        if recent_context:
            context_str = " | ".join(recent_context)
            enhanced_question = f"{enhanced_question} [Context: {context_str}]"
    
    better_question = rewrite_question_func(enhanced_question)
    print(f"Transformed query: {better_question}")
        
    return {
        "documents": documents, 
        "question": better_question,
        "transform_count": transform_count,
        "conversation_history": conversation_history,
        "question_type": question_type,
        "is_cross_document": is_cross_document
    }

def route_question(state, router_calling_func):
    """Route question to web search or RAG"""
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = router_calling_func(question)
    if source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """Determines whether to generate an answer or re-generate a question"""
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    question = state["question"]
    conversation_history = state.get("conversation_history", [])
    transform_count = state.get("transform_count", 0)
    question_type = state.get("question_type", "general")
    is_cross_document = state.get("is_cross_document", False)
    
    if transform_count >= 2:
        print("---DECISION: MAX TRANSFORMS REACHED, GENERATING ANYWAY---")
        return "generate"
    
    # For cross-document queries, be more generous
    if is_cross_document and filtered_documents:
        unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in filtered_documents)
        if len(unique_papers) >= 2:
            print(f"---DECISION: CROSS-DOCUMENT QUERY WITH {len(unique_papers)} PAPERS, GENERATING---")
            return "generate"
    
    if question_type == "title" and filtered_documents:
        print("---DECISION: TITLE QUESTION WITH DOCUMENTS, GENERATING---")
        return "generate"
    
    # Check for follow-up questions
    if not filtered_documents and conversation_history:
        follow_up_keywords = ["this", "that", "it", "which", "previous", "earlier", "them", "those"]
        is_follow_up = any(word in question.lower() for word in follow_up_keywords)
        if is_follow_up:
            print("---DECISION: FOLLOW-UP QUESTION WITH CONVERSATION CONTEXT, GENERATING---")
            return "generate"
    
    if not filtered_documents:
        print("---DECISION: NO RELEVANT DOCUMENTS, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state, hallucination_checker_func, ans_checker_func):
    """Determines whether the generation is grounded in the document and answers question"""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation_text = state["generation"]
    conversation_history = state.get("conversation_history", [])
    transform_count = state.get("transform_count", 0)
    question_type = state.get("question_type", "general")
    is_cross_document = state.get("is_cross_document", False)
    
    hallucination_retry_count = state.get("hallucination_retry_count", 0) + 1
    
    if hallucination_retry_count >= 2:
        print("---DECISION: HALLUCINATION RETRY LIMIT REACHED, ACCEPTING GENERATION---")
        return "useful"
    
    # More lenient checking for cross-document queries
    if is_cross_document:
        print("---CROSS-DOCUMENT QUERY, USING LENIENT CHECKING---")
        if any(indicator in generation_text.lower() for indicator in [
            'paper', 'papers', 'comparison', 'compare', 'versus', 'different',
            'similar', 'both', 'all', 'each', 'respectively'
        ]):
            print("---DECISION: PLAUSIBLE CROSS-DOCUMENT ANSWER, ACCEPTING---")
            return "useful"
    
    if question_type == "title":
        print("---TITLE QUESTION, USING LENIENT CHECKING---")
        if any(indicator in generation_text.lower() for indicator in [
            "research paper", "paper", "title", "detection", "deep learning", "method", "pothole", "crack"
        ]):
            print("---DECISION: PLAUSIBLE TITLE INFORMATION GENERATED, ACCEPTING---")
            return "useful"
    
    if not documents:
        print("---DECISION: NO DOCUMENTS TO CHECK AGAINST, ACCEPTING GENERATION---")
        return "useful"
    
    try:
        score = hallucination_checker_func(documents, generation_text)
        grade = score.binary_score
    except Exception as e:
        print(f"Error in hallucination checker: {e}")
        print("---DECISION: ERROR IN CHECKING, ACCEPTING GENERATION---")
        return "useful"

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        try:
            score = ans_checker_func(question, generation_text)
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        except Exception as e:
            print(f"Error in answer grading: {e}")
            return "useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        state["hallucination_retry_count"] = hallucination_retry_count
        return "not supported"
    
def classify_question_type(question):
    """Better question classification for different types of queries"""
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in [
        "title", "name of the paper", "which paper", "what paper", 
        "research paper title", "list all papers", "all titles",
        "what are the titles", "paper names", "list of research papers"
    ]):
        return "title"
    
    elif any(keyword in question_lower for keyword in [
        "author", "who wrote", "researchers", "who are the authors", "who created"
    ]):
        return "author"
    
    elif any(keyword in question_lower for keyword in [
        "method", "technique", "approach", "how did they", "how was it done",
        "algorithm", "model", "architecture"
    ]):
        return "method"
    
    elif any(keyword in question_lower for keyword in [
        "accuracy", "result", "performance", "how accurate", "what was achieved",
        "percentage", "score", "metric"
    ]):
        return "accuracy"
    
    elif any(keyword in question_lower for keyword in [
        "dataset", "data", "images", "samples", "training data"
    ]):
        return "dataset"
    
    else:
        return "general"