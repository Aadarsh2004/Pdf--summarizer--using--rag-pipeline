import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BookChat",
    page_icon="📖",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f0f0f;
    color: #e8e3da;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161616;
    border-right: 1px solid #2a2a2a;
    padding-top: 2rem;
}

/* Title */
.book-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #f0e6cc;
    letter-spacing: -0.5px;
    margin-bottom: 0;
    line-height: 1.1;
}
.book-subtitle {
    font-size: 0.85rem;
    color: #6b6b6b;
    margin-top: 4px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Chat messages */
.chat-bubble-user {
    background: #1e1e1e;
    border: 1px solid #2e2e2e;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.92rem;
    color: #e8e3da;
}
.chat-bubble-ai {
    background: #181818;
    border: 1px solid #c8a96e33;
    border-left: 3px solid #c8a96e;
    border-radius: 2px 12px 12px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 85%;
    font-size: 0.92rem;
    color: #e8e3da;
    line-height: 1.6;
}
.chat-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a4a4a;
    margin-bottom: 4px;
}

/* Upload area */
.upload-hint {
    font-size: 0.8rem;
    color: #4a4a4a;
    margin-top: 8px;
    line-height: 1.5;
}

/* Status badges */
.status-ready {
    display: inline-block;
    background: #c8a96e22;
    color: #c8a96e;
    border: 1px solid #c8a96e44;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.status-empty {
    display: inline-block;
    background: #2a2a2a;
    color: #4a4a4a;
    border: 1px solid #333;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 1.5rem 0;
}

/* Input box override */
input[type="text"], textarea {
    background-color: #1a1a1a !important;
    color: #e8e3da !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 8px !important;
}

/* Stbutton */
.stButton > button {
    background: #c8a96e;
    color: #0f0f0f;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    letter-spacing: 0.03em;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #d4b87a;
    color: #0f0f0f;
}

.stFileUploader {
    background: #161616;
    border: 1px dashed #2e2e2e;
    border-radius: 10px;
    padding: 10px;
}

/* Hide streamlit branding */
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "book_name" not in st.session_state:
    st.session_state.book_name = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ── Helpers ───────────────────────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable assistant that answers questions strictly based on the provided book content.
If the answer is not in the context, say: "I couldn't find that in the book."
Be concise, clear, and cite relevant details from the text when possible."""),
    ("human", """Context from the book:
{context}

Question: {question}""")
])


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model_name="mistral-small-latest")


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def process_pdf(uploaded_file):
    """Load PDF → split → embed → return retriever."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()

    # Use a temp directory for this session's vectorstore
    db_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=db_dir)

    os.unlink(tmp_path)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    ), len(docs), len(chunks)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="book-title">📖 BookChat</div>', unsafe_allow_html=True)
    st.markdown('<div class="book-subtitle">Chat with any PDF</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a PDF book",
        type=["pdf"],
        help="Upload any PDF — textbook, novel, paper, manual.",
    )

    if uploaded_file:
        if st.button("📥 Process Book", use_container_width=True):
            with st.spinner("Reading and indexing your book…"):
                try:
                    retriever, n_pages, n_chunks = process_pdf(uploaded_file)
                    st.session_state.retriever = retriever
                    st.session_state.book_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"Ready! {n_pages} pages → {n_chunks} chunks indexed.")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Status
    if st.session_state.book_name:
        st.markdown(f'<span class="status-ready">● Ready</span>', unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:8px;font-size:0.8rem;color:#6b6b6b;'>{st.session_state.book_name}</div>", unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-empty">○ No book loaded</span>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.retriever:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:70vh;text-align:center;gap:16px;">
        <div style="font-size:3.5rem;">📚</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#f0e6cc;">
            Upload a book to begin
        </div>
        <div style="font-size:0.9rem;color:#4a4a4a;max-width:360px;line-height:1.6;">
            Drop a PDF in the sidebar, click <strong style="color:#c8a96e;">Process Book</strong>,
            and start asking questions about its contents.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Chat header
    st.markdown(
        f'<div style="font-family:\'DM Serif Display\',serif;font-size:1.4rem;'
        f'color:#f0e6cc;margin-bottom:4px;">Chatting with</div>'
        f'<div style="font-size:0.85rem;color:#c8a96e;margin-bottom:1rem;">'
        f'{st.session_state.book_name}</div>',
        unsafe_allow_html=True
    )

    # Render chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-label">You</div>'
                    f'<div class="chat-bubble-user">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-label">BookChat</div>'
                    f'<div class="chat-bubble-ai">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )

    # Input
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask something about the book…",
            key="user_input",
            label_visibility="collapsed",
            placeholder="e.g. What are the key ideas in chapter 3?"
        )
    with col2:
        send = st.button("Send →", use_container_width=True)

    if send and user_input.strip():
        query = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("Thinking…"):
            try:
                llm = get_llm()
                docs = st.session_state.retriever.invoke(query)
                context = "\n\n".join([d.page_content for d in docs])
                final_prompt = PROMPT.invoke({"context": context, "question": query})
                response = llm.invoke(final_prompt)
                answer = response.content
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()