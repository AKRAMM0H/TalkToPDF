from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict


def load_documents(uploaded_files) -> list[Document]:
    """Extract text from uploaded PDF files."""
    documents = []
    for file in uploaded_files:
        reader = PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_num + 1, "source": file.name},
                )
            )
    return documents


def split_documents(documents: list[Document], chunk_size=1000, chunk_overlap=200) -> list[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def unique_docs(docs: list[Document]) -> list[Document]:
    """Deduplicate docs by (source, page)."""
    seen = set()
    unique = []
    for doc in docs:
        key = (doc.metadata["source"], doc.metadata["page"])
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def format_docs(docs: list[Document]) -> str:
    """Format docs into a single context string."""
    return "\n\n".join(
        f"[{doc.metadata['source']} - page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in docs
    )


def get_sources(docs: list[Document]) -> dict:
    """Return a mapping of source filename -> set of page numbers."""
    sources = defaultdict(set)
    for doc in docs:
        sources[doc.metadata["source"]].add(doc.metadata["page"])
    return sources
