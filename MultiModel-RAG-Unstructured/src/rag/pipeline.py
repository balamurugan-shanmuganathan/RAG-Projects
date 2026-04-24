from .chunks_processor import partition_document, create_chunks_by_title
from .chunks_summarizer import summarise_chunks
from .create_vector_store import create_vector_store


def run_complete_ingestion_pipeline(pdf_path: str):
    """Run the complete RAG ingestion pipeline (Synchronous version)"""
    print("🚀 Starting RAG Ingestion Pipeline")
    print("=" * 50)
    
    # Step 1: Partition
    elements = partition_document(pdf_path)
    
    # Step 2: Chunk
    chunks = create_chunks_by_title(elements)
    
    # Step 3: AI Summarisation
    summarised_chunks = summarise_chunks(chunks)
    
    # Step 4: Vector Store
    db = create_vector_store(summarised_chunks)
    
    print("🎉 Pipeline (Vector DB) completed successfully!")

def run_ingestion_with_progress(pdf_path: str):
    """Generator version of the pipeline to yield progress updates for the UI."""
    try:
        yield "EXTRACTING"
        elements = partition_document(pdf_path)
        
        yield "CHUNKING"
        chunks = create_chunks_by_title(elements)
        
        yield "SUMMARIZING"
        summarised_chunks = summarise_chunks(chunks)
        
        yield "STORING"
        create_vector_store(summarised_chunks)
        
        yield "COMPLETED"
    except Exception as e:
        yield f"ERROR: {str(e)}"

