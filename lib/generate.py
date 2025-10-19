"""
Enhanced Generate.py with Improved Cross-Document Coverage
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from lib.index_builder import get_all_titles, get_expected_titles_count
from collections import defaultdict

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

global_retriever = None

def set_global_retriever(retriever):
    global global_retriever
    global_retriever = retriever

def classify_question_intent(question: str) -> dict:
    """
    Classify question to determine if it requires cross-document analysis
    """
    question_lower = question.lower()
    
    intent = {
        'is_comparative': False,
        'is_listing': False,
        'is_aggregative': False,
        'requires_multiple_papers': False,
        'requires_all_papers': False,
        'comparison_type': None
    }
    
    # Keywords requiring ALL papers
    all_papers_keywords = [
        'all papers', 'across all', 'compare all', 'every paper',
        'all 14', 'all fourteen', 'entire collection', 'complete comparison'
    ]
    
    # Comparative keywords
    comparative_keywords = [
        'compare', 'comparison', 'versus', 'vs', 'difference', 'differences',
        'similar', 'similarity', 'better', 'best', 'worst', 'higher', 'lower',
        'which paper', 'which method', 'which approach', 'highest', 'lowest'
    ]
    
    # Listing keywords
    listing_keywords = [
        'all', 'list', 'enumerate', 'what are the', 'how many',
        'titles', 'papers', 'methods', 'techniques', 'approaches'
    ]
    
    # Aggregative keywords
    aggregative_keywords = [
        'common', 'commonly', 'typical', 'generally', 'usually',
        'most', 'least', 'average', 'across papers', 'in all papers'
    ]
    
    # Check if query requires ALL papers
    if any(keyword in question_lower for keyword in all_papers_keywords):
        intent['requires_all_papers'] = True
        intent['requires_multiple_papers'] = True
    
    if any(keyword in question_lower for keyword in comparative_keywords):
        intent['is_comparative'] = True
        intent['requires_multiple_papers'] = True
        
        if 'accuracy' in question_lower or 'performance' in question_lower:
            intent['comparison_type'] = 'accuracy'
        elif 'method' in question_lower or 'technique' in question_lower:
            intent['comparison_type'] = 'method'
        elif 'dataset' in question_lower:
            intent['comparison_type'] = 'dataset'
        else:
            intent['comparison_type'] = 'general'
    
    if any(keyword in question_lower for keyword in listing_keywords):
        intent['is_listing'] = True
        intent['requires_multiple_papers'] = True
    
    if any(keyword in question_lower for keyword in aggregative_keywords):
        intent['is_aggregative'] = True
        intent['requires_multiple_papers'] = True
    
    return intent

def get_comprehensive_context(retriever, question, num_papers=14):
    """
    Retrieve context from ALL papers for comprehensive analysis
    """
    if not retriever:
        return []
    
    # Get all paper titles
    all_titles = retriever.get_all_paper_titles()
    
    # Search across all papers
    results = retriever.search_across_papers(question, paper_titles=all_titles)
    
    # Convert to list of documents
    all_docs = []
    for title, chunks in results.items():
        all_docs.extend(chunks)
    
    # If we don't have enough coverage, do additional retrieval
    if len(set(doc.metadata.get('title', 'Unknown') for doc in all_docs)) < 10:
        # Do a broader search
        additional_docs = retriever.invoke(question, cross_document=True)
        
        # Merge results
        existing_ids = set(id(doc) for doc in all_docs)
        for doc in additional_docs:
            if id(doc) not in existing_ids:
                all_docs.append(doc)
    
    return all_docs

def format_docs_by_paper(docs):
    """Group documents by paper title - using FULL titles only"""
    if not docs:
        return "No documents available."
    
    papers = defaultdict(list)
    
    for doc in docs:
        title = doc.metadata.get('title', 'Unknown Paper')
        papers[title].append(doc.page_content)
    
    formatted = []
    for title, contents in papers.items():
        formatted.append(f"\n=== {title} ===\n")
        formatted.append("\n".join(contents))
    
    return "\n".join(formatted)

def format_docs(docs):
    """Simple format for single-paper queries"""
    if not docs:
        return "No documents available."
    return "\n\n".join(doc.page_content for doc in docs)

def generate_comparative_answer(docs, question, question_intent):
    """Generate answer for comparative questions across papers"""
    
    comparison_type = question_intent.get('comparison_type', 'general')
    
    # Count unique papers in context
    unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in docs)
    num_papers = len(unique_papers)
    
    prompt_templates = {
        'accuracy': f"""You are comparing accuracies and performance metrics across ALL 14 research papers in our database.

Context from {num_papers} papers:
{{context}}

Question: {{question}}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:
1. ALWAYS use the FULL paper titles as shown in the context
2. NEVER refer to papers as "Paper 1", "Paper 2", etc.
3. Extract accuracy/performance metrics from EVERY SINGLE paper mentioned in the context
4. Present them in a numbered list format with FULL paper titles
5. YOU MUST AIM FOR ALL 14 PAPERS - if you only see {num_papers} papers in context, list ALL of them
6. Highlight the best and worst performers by their FULL titles
7. If any paper lacks accuracy information, still mention it and state "accuracy not specified"
8. Count how many papers you've listed and state the total at the end

MANDATORY FORMAT:
"Accuracy Comparison Across All Papers:

1. [Full Paper Title] - X% accuracy [brief method if mentioned]
2. [Full Paper Title] - Y% accuracy [brief method if mentioned]
...
[Continue for EVERY paper in the context]

Summary:
- Highest accuracy: [Paper name] with X%
- Lowest accuracy: [Paper name] with Y%
- Total papers analyzed: [NUMBER]"

Answer with comprehensive comparison covering ALL {num_papers} papers:""",
        
        'method': f"""You are comparing methodologies and techniques across ALL 14 research papers.

Context from {num_papers} papers:
{{context}}

Question: {{question}}

CRITICAL INSTRUCTIONS - LIST EVERY PAPER:
1. ALWAYS use the FULL paper titles as shown in the context
2. NEVER use "Paper 1", "Paper 2" - always use complete paper titles
3. Identify the key methods/techniques used in EACH of the {num_papers} papers
4. YOU MUST LIST ALL {num_papers} PAPERS - do not skip any
5. Group similar methods together if helpful, but list ALL papers
6. Mention unique innovations with full paper titles
7. State the total number of papers covered at the end

MANDATORY FORMAT:
"Method Comparison Across All Papers:

Deep Learning-Based Methods:
1. [Full Paper Title] - [Method details]
2. [Full Paper Title] - [Method details]
...

Traditional Methods:
1. [Full Paper Title] - [Method details]
...

Hybrid Approaches:
...

Total papers analyzed: {num_papers}"

Answer covering ALL {num_papers} papers:""",
        
        'dataset': f"""You are comparing datasets used across ALL 14 research papers.

Context from {num_papers} papers:
{{context}}

Question: {{question}}

CRITICAL INSTRUCTIONS - COVER ALL PAPERS:
1. ALWAYS refer to papers by their FULL titles as shown in the context
2. NEVER use "Paper 1", "Paper 2" abbreviations
3. List the datasets used in EACH of the {num_papers} papers
4. YOU MUST MENTION ALL {num_papers} PAPERS
5. Include dataset sizes when mentioned
6. Compare dataset types and characteristics
7. Note which papers used similar/same datasets
8. If dataset info is not available for a paper, state "dataset not specified"
9. Count and state total papers at the end

MANDATORY FORMAT:
"Dataset Comparison Across All Papers:

Large Datasets (>10,000 samples):
1. [Full Paper Title]: [dataset details]
...

Medium Datasets (1,000-10,000):
1. [Full Paper Title]: [dataset details]
...

Small Datasets (<1,000):
...

Papers without specified dataset size:
...

Total papers analyzed: {num_papers}"

Answer covering ALL {num_papers} papers:""",
        
        'general': f"""You are comparing information across ALL 14 research papers available.

Context from {num_papers} papers:
{{context}}

Question: {{question}}

CRITICAL INSTRUCTIONS - COMPREHENSIVE COVERAGE REQUIRED:
1. ALWAYS use FULL paper titles exactly as they appear in the context
2. NEVER abbreviate to "Paper 1", "Paper 2", "Paper X"
3. Analyze information from EVERY SINGLE ONE of the {num_papers} papers in the context
4. YOU MUST LIST ALL {num_papers} PAPERS - this is mandatory
5. Present a clear, structured comparison covering ALL papers
6. Highlight key similarities and differences
7. Use complete paper titles to reference specific findings
8. Number your list and count total papers at the end

MANDATORY: State "Total papers analyzed: {num_papers}" at the end of your response.

Answer covering ALL {num_papers} papers:"""
    }
    
    prompt_text = prompt_templates.get(comparison_type, prompt_templates['general'])
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    rag_chain = prompt | llm | StrOutputParser()
    
    formatted_context = format_docs_by_paper(docs)
    
    answer = rag_chain.invoke({
        "context": formatted_context,
        "question": question
    })
    
    # Post-process to ensure paper count is mentioned
    if "Total papers analyzed:" not in answer:
        answer += f"\n\nTotal papers analyzed: {num_papers}"
    
    return answer

def generate_aggregative_answer(docs, question):
    """Generate answer that aggregates information across papers"""
    
    # Count unique papers
    unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in docs)
    num_papers = len(unique_papers)
    
    prompt_text = f"""You are analyzing common patterns, trends, or statistics across ALL 14 research papers.

Context from {num_papers} papers:
{{context}}

Question: {{question}}

CRITICAL INSTRUCTIONS - COMPLETE COVERAGE REQUIRED:
1. ALWAYS use FULL paper titles as shown in the context
2. NEVER refer to papers as "Paper 1", "Paper 2", etc.
3. Identify patterns or common elements across ALL {num_papers} papers in the context
4. Provide aggregate statistics or summaries where relevant
5. Note exceptions or outliers
6. Reference specific papers by their COMPLETE titles when making claims
7. YOU MUST LIST EVERY PAPER that is relevant to the question
8. State the total number of papers analyzed at the end

If the question asks for counts or statistics, provide specific numbers with full paper titles.
If asking about common practices, list them with supporting evidence using complete paper names.

MANDATORY FORMAT:
"[Analysis Topic]:

Category 1 (used in X papers):
1. [Full Paper Title] - [details]
2. [Full Paper Title] - [details]
...

Category 2 (used in Y papers):
1. [Full Paper Title] - [details]
...

Summary:
- Total papers analyzed: {num_papers}
- Most common approach: [details]"

Answer covering ALL {num_papers} papers:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    rag_chain = prompt | llm | StrOutputParser()
    
    formatted_context = format_docs_by_paper(docs)
    
    answer = rag_chain.invoke({
        "context": formatted_context,
        "question": question
    })
    
    # Ensure paper count is mentioned
    if "Total papers analyzed:" not in answer:
        answer += f"\n\nTotal papers analyzed: {num_papers}"
    
    return answer

def generation(docs, question, conversation_history=None, question_type="general"):
    """
    Enhanced generation with improved cross-document analysis support
    """
    try:
        # Classify question intent
        intent = classify_question_intent(question)
        
        # If query requires all papers and we have retriever, get comprehensive context
        if intent['requires_all_papers'] and global_retriever:
            print("Query requires ALL papers - retrieving comprehensive context")
            comprehensive_docs = get_comprehensive_context(global_retriever, question)
            
            # Merge with existing docs
            all_doc_ids = set()
            merged_docs = []
            
            for doc in docs + comprehensive_docs:
                doc_id = (doc.metadata.get('title', ''), doc.page_content[:100])
                if doc_id not in all_doc_ids:
                    merged_docs.append(doc)
                    all_doc_ids.add(doc_id)
            
            docs = merged_docs
            
            unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in docs)
            print(f"Retrieved context from {len(unique_papers)} papers")
        
        # Handle title listing questions
        if question_type == "title" or "title" in question.lower():
            if "compare" in question.lower() or "which" in question.lower():
                pass  # Will be handled below
            else:
                return get_all_titles_manually()
        
        # Handle comparative questions
        if intent['is_comparative'] and docs:
            return generate_comparative_answer(docs, question, intent)
        
        # Handle aggregative questions
        if intent['is_aggregative'] and docs:
            return generate_aggregative_answer(docs, question)
        
        # Handle listing questions with cross-document context
        if intent['is_listing'] and intent['requires_multiple_papers']:
            if not docs:
                return "I don't have enough information to answer this question comprehensively."
            
            # Count unique papers
            unique_papers = set(doc.metadata.get('title', 'Unknown') for doc in docs)
            num_papers = len(unique_papers)
            
            prompt_text = f"""You are providing a comprehensive list or summary across ALL research papers.

Context from {num_papers} papers:
{{context}}

Question: {{question}}

CRITICAL INSTRUCTIONS:
1. ALWAYS use FULL paper titles as shown in the context
2. NEVER use "Paper 1", "Paper 2" abbreviations
3. Extract relevant information from ALL {num_papers} papers mentioned
4. Present in a clear list or structured format
5. Organize by paper using complete paper titles
6. Be thorough and include ALL relevant findings
7. Number your list for clarity
8. State total papers at the end

Example format:
"Papers Using [Topic]:

1. [Full Paper Title]
   - [Details]

2. [Full Paper Title]
   - [Details]

[Continue for ALL relevant papers]

Total papers analyzed: {num_papers}"

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(prompt_text)
            rag_chain = prompt | llm | StrOutputParser()
            
            answer = rag_chain.invoke({
                "context": format_docs_by_paper(docs),
                "question": question
            })
            
            # Ensure paper count is mentioned
            if "Total papers analyzed:" not in answer:
                answer += f"\n\nTotal papers analyzed: {num_papers}"
            
            return answer
        
        # Standard single-focus questions
        if docs:
            # Check if multiple papers are present
            unique_titles = set(doc.metadata.get('title', 'Unknown') for doc in docs)
            
            if len(unique_titles) > 1:
                context_format = format_docs_by_paper(docs)
                context_note = "\n\nNote: Information from multiple papers is provided above. Always refer to papers by their full titles."
            else:
                context_format = format_docs(docs)
                context_note = ""
            
            prompt_text = f"""Use the following context to answer the question. 

IMPORTANT: If information comes from multiple papers, ALWAYS use their FULL titles, NEVER abbreviate to "Paper 1" or "Paper 2".

Keep the answer clear and concise.{context_note}

Context: {{context}}
Question: {{question}}
Answer:"""
            
            prompt = ChatPromptTemplate.from_template(prompt_text)
            rag_chain = prompt | llm | StrOutputParser()
            
            return rag_chain.invoke({
                "context": context_format,
                "question": question
            })
        
        return "I couldn't find relevant information to answer your question. Please try rephrasing or asking about specific papers."
    
    except Exception as e:
        print(f"Generation error: {e}")
        return "I encountered an error while generating an answer. Please try again."

def get_all_titles_manually():
    """Get ALL titles directly from the manual mapping"""
    try:
        titles = get_all_titles()
        expected_count = get_expected_titles_count()
        
        if not titles:
            return "No titles found in the manual mapping."
        
        response = "Research Paper Titles\n\n"
        for i, title in enumerate(titles, 1):
            response += f"{i}. {title}\n"
        
        response += f"\nDatabase Summary\n"
        response += f"- Total titles found: {len(titles)}\n"
        response += f"- Expected papers: {expected_count}\n"
        
        if len(titles) < expected_count:
            missing_count = expected_count - len(titles)
            response += f"\nNote: Found {len(titles)} out of {expected_count} expected titles.\n"
            response += f"Missing {missing_count} papers - they may not have been processed successfully.\n"
        
        return response
        
    except Exception as e:
        return f"Error retrieving titles: {str(e)}"