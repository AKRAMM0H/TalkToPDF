import streamlit as st
from dotenv import load_dotenv

from utils import load_documents, split_documents, get_sources
from retriever import build_retriever
from chain import build_chain

load_dotenv()

# ── Session State ──────────────────────────────────────────────────────────────
for key, default in {
    "vectorstore": None,
    "chunks": None,
    "retriever": None,
    "store": {},
    "messages": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📄 TalkToPDF")
st.subheader("Chat with your PDFs")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# ── Process PDFs ───────────────────────────────────────────────────────────────
if uploaded_files and st.button("Process PDFs"):
    with st.spinner("Processing PDFs..."):
        documents = load_documents(uploaded_files)
        chunks = split_documents(documents)
        st.session_state.chunks = chunks

        vectorstore, retriever = build_retriever(chunks)
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = retriever

    st.success(f"Loaded {len(chunks)} chunks from {len(uploaded_files)} PDFs 🎉")

# ── Chat History ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            for file, pages in msg["sources"].items():
                pages_str = ", ".join(map(str, sorted(pages)))
                st.info(f"📄 {file} — Pages {pages_str}")

# ── Chat Input ─────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your PDFs...")

if question:
    if st.session_state.retriever is None:
        st.warning("Please upload and process PDFs first.")
        st.stop()

    with st.chat_message("human"):
        st.write(question)
    st.session_state.messages.append({"role": "human", "content": question})

    chain = build_chain(st.session_state.retriever, st.session_state.store)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": "user"}},
            )

        answer = result["answer"]
        sources = get_sources(result["docs"])

        st.write(answer)
        if sources:
            st.markdown("### 📄 Sources")
            for file, pages in sources.items():
                pages_str = ", ".join(map(str, sorted(pages)))
                st.info(f"{file} — Pages {pages_str}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
