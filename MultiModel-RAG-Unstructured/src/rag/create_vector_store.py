from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ..utils.config import EMBEDDING_MODEL, VECTOR_STORE_PATH
 
def create_vector_store(documents):
    """Create and persist ChromaDB vector store"""
    print("🔮 Creating embeddings and storing in ChromaDB...")
        
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=VECTOR_STORE_PATH, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"✅ Vector store created and saved to {VECTOR_STORE_PATH}")
    return vectorstore