import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from utils import unique_docs, format_docs


def get_session_history(session_id: str, store: dict) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_chain(retriever, store: dict):
    """Build the RAG chain with conversation memory."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI assistant. Use the context below to answer. "
            "If the answer is not in the context, say you don't know.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    rag_chain = (
        RunnablePassthrough.assign(
            docs=RunnableLambda(lambda x: x["question"])
            | retriever
            | RunnableLambda(unique_docs)
        )
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RunnablePassthrough.assign(answer=prompt | llm | StrOutputParser())
    )

    chain_with_memory = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: get_session_history(session_id, store),
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_memory
