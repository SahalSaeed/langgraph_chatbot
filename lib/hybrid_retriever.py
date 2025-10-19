"""
Enhanced Hybrid Retriever with Improved Cross-Document Analysis Support
"""
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Optional
from langchain.schema import Document
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnhancedHybridRetriever:
    """
    Two-Level Retrieval with Cross-Document Analysis Capabilities
    """
    
    def __init__(self, vectorstore: Chroma, documents: List[Document], weights: List[float] = None):
        """
        Initialize enhanced hybrid retriever with cross-document capabilities
        
        Args:
            vectorstore: Chroma vectorstore for dense retrieval
            documents: All documents for BM25 sparse retrieval
            weights: [sparse_weight, dense_weight], defaults to [0.4, 0.6]
        """
        self.vectorstore = vectorstore
        self.documents = documents
        self.weights = weights or [0.4, 0.6]
        
        # Build document index for quick access
        self.doc_index = self._build_document_index(documents)
        
        # Configure retrievers
        self.dense_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}  # Increased for better coverage
        )
        
        self.sparse_retriever = BM25Retriever.from_documents(documents)
        self.sparse_retriever.k = 15
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.sparse_retriever, self.dense_retriever],
            weights=self.weights
        )
        
        logger.info(f"Enhanced hybrid retriever initialized with {len(self.doc_index)} unique papers")
        logger.info(f"Weights: sparse={self.weights[0]}, dense={self.weights[1]}")
    
    def _build_document_index(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Build an index mapping paper titles to their chunks"""
        index = defaultdict(list)
        
        for doc in documents:
            title = doc.metadata.get('title', 'Unknown')
            index[title].append(doc)
        
        logger.info(f"Built document index with {len(index)} unique papers")
        return dict(index)
    
    def get_all_paper_titles(self) -> List[str]:
        """Get list of all paper titles in the database"""
        return list(self.doc_index.keys())
    
    def get_paper_chunks(self, title: str) -> List[Document]:
        """Get all chunks for a specific paper"""
        return self.doc_index.get(title, [])
    
    def get_papers_by_topic(self, topic: str) -> List[str]:
        """Get papers related to a specific topic"""
        matching_papers = []
        topic_lower = topic.lower()
        
        for title, chunks in self.doc_index.items():
            # Check if topic appears in paper metadata or content
            if any(topic_lower in chunk.metadata.get('topics', '').lower() 
                   for chunk in chunks):
                matching_papers.append(title)
            elif any(topic_lower in chunk.page_content.lower() 
                    for chunk in chunks[:3]):  # Check first 3 chunks
                matching_papers.append(title)
        
        return matching_papers
    
    def invoke(self, query: str, cross_document: bool = False) -> List[Document]:
        """
        Retrieve documents using hybrid approach
        
        Args:
            query: Search query
            cross_document: If True, ensures results span multiple papers
        """
        try:
            results = self.ensemble_retriever.invoke(query)
            
            if cross_document:
                results = self._diversify_results(results, max_per_paper=3)
            
            logger.info(f"Hybrid retrieval returned {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Hybrid retrieval error: {e}, falling back to dense only")
            return self.dense_retriever.invoke(query)
    
    def _diversify_results(self, results: List[Document], max_per_paper: int = 3) -> List[Document]:
        """
        Ensure results span multiple papers for cross-document analysis
        
        Args:
            results: Retrieved documents
            max_per_paper: Maximum chunks per paper
        """
        paper_counts = defaultdict(int)
        diversified = []
        
        for doc in results:
            title = doc.metadata.get('title', 'Unknown')
            
            if paper_counts[title] < max_per_paper:
                diversified.append(doc)
                paper_counts[title] += 1
        
        # If we need more documents, add remaining ones
        remaining = [doc for doc in results if doc not in diversified]
        diversified.extend(remaining[:max(0, 15 - len(diversified))])
        
        logger.info(f"Diversified results across {len(paper_counts)} papers")
        return diversified
    
    def get_comparative_context(self, query: str, num_papers: int = 5) -> Dict[str, List[Document]]:
        """
        Get context from multiple papers for comparative analysis
        
        Args:
            query: Search query
            num_papers: Number of papers to include
            
        Returns:
            Dict mapping paper titles to their relevant chunks
        """
        results = self.invoke(query, cross_document=True)
        
        paper_chunks = defaultdict(list)
        
        for doc in results:
            title = doc.metadata.get('title', 'Unknown')
            paper_chunks[title].append(doc)
            
            if len(paper_chunks) >= num_papers:
                break
        
        logger.info(f"Retrieved comparative context from {len(paper_chunks)} papers")
        return dict(paper_chunks)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Alternative method name for compatibility"""
        return self.invoke(query)
    
    def search_across_papers(self, query: str, paper_titles: List[str] = None) -> Dict[str, List[Document]]:
        """
        Search specific papers or all papers with improved relevance
        
        Args:
            query: Search query
            paper_titles: Optional list of paper titles to search. If None, searches all papers.
            
        Returns:
            Dict mapping paper titles to matching chunks
        """
        if paper_titles is None:
            paper_titles = self.get_all_paper_titles()
        
        results = {}
        query_terms = set(query.lower().split())
        
        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_terms = query_terms - stop_words
        
        for title in paper_titles:
            paper_chunks = self.get_paper_chunks(title)
            
            # Score each chunk based on query term matches
            scored_chunks = []
            for chunk in paper_chunks:
                content_lower = chunk.page_content.lower()
                
                # Count matches
                matches = sum(1 for term in query_terms if term in content_lower)
                
                # Also check metadata
                metadata_matches = 0
                for key, value in chunk.metadata.items():
                    if isinstance(value, str) and any(term in value.lower() for term in query_terms):
                        metadata_matches += 1
                
                total_score = matches + (metadata_matches * 2)  # Weight metadata higher
                
                if total_score > 0:
                    scored_chunks.append((chunk, total_score))
            
            # Sort by score and take top chunks
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            if scored_chunks:
                # Take top 2-3 chunks per paper
                results[title] = [chunk for chunk, score in scored_chunks[:3]]
        
        logger.info(f"Cross-paper search found results in {len(results)} papers")
        return results
    
    def get_all_papers_sample(self, chunks_per_paper: int = 2) -> List[Document]:
        """
        Get a representative sample from ALL papers
        
        Args:
            chunks_per_paper: Number of chunks to get from each paper
            
        Returns:
            List of documents with representation from all papers
        """
        sample_docs = []
        
        for title, chunks in self.doc_index.items():
            # Get first N chunks from each paper
            sample_docs.extend(chunks[:chunks_per_paper])
        
        logger.info(f"Retrieved {len(sample_docs)} chunks from {len(self.doc_index)} papers")
        return sample_docs