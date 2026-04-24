from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ..utils.config import EMBEDDING_MODEL, VECTOR_STORE_PATH


def get_retriever(search_type="similarity", k=5):
    """
    Load persisted ChromaDB and return retriever
    """

    print("🔍 Loading vector store retriever...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )

    print("✅ Retriever loaded successfully")

    return retriever
