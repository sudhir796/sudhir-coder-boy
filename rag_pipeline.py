"""
Retrieval-Augmented Generation (RAG) Pipeline
Implements all 17 steps from RAG_Step_by_Step-2.ipynb
Supports: Document chunking, embeddings, similarity search, LangChain PDF loading, FAISS vector database
"""

import logging
import sys
import argparse
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    'embedding_model': 'all-MiniLM-L6-v2',  # Sentence Transformer model
    'chunk_size': 30,  # For LangChain text splitter
    'chunk_overlap': 2,  # For LangChain text splitter
    'similarity_threshold': 0.5,  # Minimum similarity score
}


# ==============================================================================
# STEP 1-3: DOCUMENT AND CHUNKING
# ==============================================================================

def create_document(text: str) -> str:
    """
    Step 1: Create/define a document
    
    Args:
        text: Raw document text
    
    Returns:
        The formatted document string
    """
    logger.info("Step 1: Creating document")
    logger.debug(f"Document length: {len(text)} characters")
    return text


def chunk_document(text: str, separator: str = ".") -> List[str]:
    """
    Step 2-3: Split document into chunks and remove empty ones
    
    Args:
        text: Document text to chunk
        separator: Character to split on (default: period)
    
    Returns:
        List of non-empty chunks
    """
    logger.info("Step 2-3: Chunking document and removing empty chunks")
    
    # Split by separator
    chunks = text.split(separator)
    
    # Remove empty/whitespace-only chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    logger.info(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        logger.debug(f"  Chunk {i}: {chunk[:50]}...")
    
    return chunks


# ==============================================================================
# STEP 5-8: EMBEDDINGS
# ==============================================================================

def load_embedding_model():
    """
    Step 4: Install and load Sentence Transformer model
    
    Returns:
        SentenceTransformer model instance
    """
    logger.info(f"Step 4-5: Loading embedding model: {CONFIG['embedding_model']}")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(CONFIG['embedding_model'])
        logger.info("Embedding model loaded successfully")
        return model
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)


def encode_chunks(model, chunks: List[str]) -> np.ndarray:
    """
    Step 6: Encode multiple chunks into embeddings
    
    Args:
        model: Sentence Transformer model
        chunks: List of text chunks
    
    Returns:
        Numpy array of embeddings (shape: [num_chunks, embedding_dim])
    """
    logger.info(f"Step 6: Encoding {len(chunks)} chunks")
    embeddings = model.encode(chunks)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def encode_query(model, query: str) -> np.ndarray:
    """
    Step 7-8: Encode user query into embedding
    
    Args:
        model: Sentence Transformer model
        query: User query text
    
    Returns:
        Numpy array of query embedding
    """
    logger.info(f"Step 7-8: Encoding query: '{query}'")
    query_embedding = model.encode(query)
    logger.info(f"Query embedding shape: {query_embedding.shape}")
    return query_embedding


# ==============================================================================
# STEP 10-12: SIMILARITY SEARCH AND PROMPT AUGMENTATION
# ==============================================================================

def similarity_search(query_embedding: np.ndarray, 
                     chunk_embeddings: np.ndarray) -> Tuple[int, float]:
    """
    Step 10: Calculate cosine similarity between query and chunks
    
    Args:
        query_embedding: Query embedding vector
        chunk_embeddings: Chunk embeddings matrix
    
    Returns:
        Tuple of (best_match_index, highest_similarity_score)
    """
    logger.info("Step 10: Performing similarity search")
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        best_index = np.argmax(similarities)
        best_score = similarities[best_index]
        logger.info(f"Similarities: {similarities}")
        logger.info(f"Best match index: {best_index}, Score: {best_score:.4f}")
        return best_index, best_score
    except ImportError:
        logger.error("scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)


def retrieve_best_chunk(chunks: List[str], best_index: int) -> str:
    """
    Step 11: Retrieve best matching chunk
    
    Args:
        chunks: List of all chunks
        best_index: Index of best matching chunk
    
    Returns:
        Best matching chunk text
    """
    logger.info("Step 11: Retrieving best matching chunk")
    retrieved_chunk = chunks[best_index]
    logger.info(f"Retrieved chunk: {retrieved_chunk}")
    return retrieved_chunk


def augment_prompt(context: str, query: str) -> str:
    """
    Step 12: Create augmented prompt with context
    
    Args:
        context: Retrieved context chunk
        query: User query
    
    Returns:
        Augmented prompt string
    """
    logger.info("Step 12: Augmenting prompt with context")
    prompt = f'''
Context:
{context}

Question:
{query}
'''
    logger.info(f"Augmented prompt created:\n{prompt}")
    return prompt


# ==============================================================================
# STEP 14: COMPLETE MINI RAG PIPELINE
# ==============================================================================

def rag_pipeline_basic(document: str, query: str, model=None) -> str:
    """
    Step 14: Complete mini RAG pipeline (Steps 1-14)
    Combines all steps: document -> chunks -> embeddings -> similarity search -> augmented prompt
    
    Args:
        document: Raw document text
        query: User query
        model: Optional pre-loaded embedding model (if None, will load)
    
    Returns:
        Augmented prompt string
    """
    logger.info("=" * 80)
    logger.info("STEP 14: COMPLETE MINI RAG PIPELINE (Steps 1-14)")
    logger.info("=" * 80)
    
    # Load model if not provided
    if model is None:
        model = load_embedding_model()
    
    # Step 1-3: Create document and chunks
    doc = create_document(document)
    chunks = chunk_document(doc)
    
    # Step 6: Encode chunks
    chunk_embeddings = encode_chunks(model, chunks)
    
    # Step 7-8: Encode query
    query_embedding = encode_query(model, query)
    
    # Step 10-11: Find best matching chunk
    best_index, similarity_score = similarity_search(query_embedding, chunk_embeddings)
    retrieved_chunk = retrieve_best_chunk(chunks, best_index)
    
    # Step 12: Augment prompt
    augmented_prompt = augment_prompt(retrieved_chunk, query)
    
    logger.info("=" * 80)
    logger.info(f"MINI RAG COMPLETE - Similarity Score: {similarity_score:.4f}")
    logger.info("=" * 80)
    
    return augmented_prompt


# ==============================================================================
# STEP 15: LANGCHAIN DOCUMENT LOADING AND CHUNKING
# ==============================================================================

def load_pdf_document(pdf_path: str) -> List[str]:
    """
    Step 15: Load PDF document using PyPDFLoader
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        List of document chunks
    """
    logger.info(f"Step 15: Loading PDF from {pdf_path}")
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        # Extract text from documents
        text_content = "\n".join([doc.page_content for doc in documents])
        logger.info(f"Total text length: {len(text_content)} characters")
        
        return text_content
    except ImportError:
        logger.error("langchain-community not installed. Run: pip install langchain-community langchain-text-splitters")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)


def chunk_text_langchain(text: str, chunk_size: Optional[int] = None, 
                        chunk_overlap: Optional[int] = None) -> List[str]:
    """
    Step 15 (advanced): Use LangChain CharacterTextSplitter for chunking
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (uses CONFIG default if None)
        chunk_overlap: Overlap between chunks (uses CONFIG default if None)
    
    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = CONFIG['chunk_size']
    if chunk_overlap is None:
        chunk_overlap = CONFIG['chunk_overlap']
    
    logger.info(f"Step 15: Chunking text using LangChain (size={chunk_size}, overlap={chunk_overlap})")
    try:
        from langchain_text_splitters import CharacterTextSplitter
        
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks using CharacterTextSplitter")
        return chunks
    except ImportError:
        logger.error("langchain-text-splitters not installed. Run: pip install langchain-text-splitters")
        sys.exit(1)


# ==============================================================================
# STEP 16-17: FAISS VECTOR DATABASE
# ==============================================================================

def create_faiss_vector_store(chunks: List[str], model=None):
    """
    Step 16: Create FAISS vector database from chunks
    
    Args:
        chunks: List of text chunks
        model: Optional pre-loaded embedding model
    
    Returns:
        FAISS vector store object
    """
    logger.info("Step 16: Creating FAISS vector database")
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Use HuggingFace embeddings
        embedding_model = HuggingFaceEmbeddings(model_name=CONFIG['embedding_model'])
        logger.info(f"Creating FAISS DB with {len(chunks)} chunks")
        
        # Create FAISS vector store
        db = FAISS.from_texts(chunks, embedding_model)
        logger.info("FAISS vector database created successfully")
        return db
    except ImportError:
        logger.error("Required packages not installed. Run: pip install langchain-community langchain-huggingface faiss-cpu")
        sys.exit(1)


def search_faiss_vector_store(vector_store, query: str, k: int = 1) -> List[Tuple[str, float]]:
    """
    Step 17: Retrieve relevant chunks from FAISS using similarity search
    
    Args:
        vector_store: FAISS vector store object
        query: User query
        k: Number of results to return
    
    Returns:
        List of tuples (chunk_text, similarity_score)
    """
    logger.info(f"Step 17: Searching FAISS vector store with query: '{query}'")
    
    try:
        # Perform similarity search
        results = vector_store.similarity_search_with_scores(query, k=k)
        
        logger.info(f"Found {len(results)} results")
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"  Result {i}: Score={score:.4f}, Content={doc.page_content[:50]}...")
        
        # Return formatted results
        return [(doc.page_content, score) for doc, score in results]
    except Exception as e:
        logger.error(f"Error searching FAISS: {e}")
        return []


# ==============================================================================
# END-TO-END PIPELINE WITH FAISS
# ==============================================================================

def rag_pipeline_faiss(document: str, query: str, use_langchain_splitter: bool = False) -> str:
    """
    Complete RAG pipeline with FAISS vector database (Steps 15-17)
    
    Args:
        document: Document text or PDF path
        query: User query
        use_langchain_splitter: Whether to use LangChain's CharacterTextSplitter
    
    Returns:
        Augmented prompt string with best retrieved context
    """
    logger.info("=" * 80)
    logger.info("COMPLETE RAG PIPELINE WITH FAISS (Steps 15-17)")
    logger.info("=" * 80)
    
    # Step 15: Chunk the document
    if use_langchain_splitter:
        chunks = chunk_text_langchain(document)
    else:
        chunks = chunk_document(document)
    
    # Step 16: Create FAISS vector store
    vector_store = create_faiss_vector_store(chunks)
    
    # Step 17: Search FAISS and retrieve best chunk
    results = search_faiss_vector_store(vector_store, query, k=1)
    
    if results:
        retrieved_chunk, similarity_score = results[0]
        logger.info(f"Best match similarity score: {similarity_score:.4f}")
    else:
        logger.warning("No results found, using first chunk as fallback")
        retrieved_chunk = chunks[0]
    
    # Augment prompt
    augmented_prompt = augment_prompt(retrieved_chunk, query)
    
    logger.info("=" * 80)
    logger.info("FAISS RAG PIPELINE COMPLETE")
    logger.info("=" * 80)
    
    return augmented_prompt


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_results(augmented_prompt: str, similarity_score: Optional[float] = None):
    """Pretty print results"""
    print("\n" + "=" * 80)
    print("GENERATED AUGMENTED PROMPT:")
    print("=" * 80)
    print(augmented_prompt)
    if similarity_score:
        print(f"\nSimilarity Score: {similarity_score:.4f}")
    print("=" * 80 + "\n")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Retrieval-Augmented Generation (RAG) Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic mini RAG pipeline with text input
  python rag_pipeline.py --mode basic --document "AI is transforming education. RAG helps reduce hallucination." --query "What reduces hallucination?"
  
  # FAISS pipeline with PDF input
  python rag_pipeline.py --mode faiss --pdf /path/to/document.pdf --query "What is your question?"
  
  # FAISS pipeline with LangChain chunking
  python rag_pipeline.py --mode faiss --document "Your document text" --query "Your query" --use-langchain
        """
    )
    
    parser.add_argument('--mode', choices=['basic', 'faiss'], default='basic',
                       help='RAG pipeline mode: basic (cosine similarity) or faiss (vector DB)')
    parser.add_argument('--document', type=str, default=None,
                       help='Document text to process')
    parser.add_argument('--pdf', type=str, default=None,
                       help='Path to PDF file to process')
    parser.add_argument('--query', type=str, required=True,
                       help='Query to search for (required)')
    parser.add_argument('--use-langchain', action='store_true',
                       help='Use LangChain CharacterTextSplitter (for FAISS mode)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine input document
    if args.pdf:
        document_text = load_pdf_document(args.pdf)
    elif args.document:
        document_text = args.document
    else:
        # Default example if no input provided
        document_text = '''
AI is transforming education.
RAG helps reduce hallucination in LLMs.
Embeddings convert text into vectors.
Vector databases enable efficient similarity search.
'''
        logger.info("No input provided, using default example document")
    
    logger.info(f"Processing document (length: {len(document_text)} chars)")
    logger.info(f"Query: {args.query}")
    
    try:
        # Run selected pipeline
        if args.mode == 'basic':
            model = load_embedding_model()
            augmented_prompt = rag_pipeline_basic(document_text, args.query, model)
            print_results(augmented_prompt)
        
        elif args.mode == 'faiss':
            augmented_prompt = rag_pipeline_faiss(
                document_text, 
                args.query, 
                use_langchain_splitter=args.use_langchain
            )
            print_results(augmented_prompt)
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
