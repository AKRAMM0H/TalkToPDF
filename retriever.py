from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_retriever(chunks: list[Document]):
    """
    Build a hybrid retriever:
      - Semantic search via ChromaDB + HuggingFace embeddings
      - Keyword search via BM25
      - Re-ranked with FlashRank
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(chunks, embeddings)

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 10

    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25],
        weights=[0.7, 0.3],
    )

    FlashrankRerank.model_rebuild()
    compressor = FlashrankRerank(top_n=4)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble,
    )

    return vectorstore, retriever
