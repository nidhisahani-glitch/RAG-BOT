import os
import tempfile
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from datetime import datetime
from collections import Counter
import toml

# Load API key from config.toml
config = toml.load("config.toml")
together_api_key = config["together"]["api_key"]

if not together_api_key:
    raise ValueError("Together API key not found. Make sure config.toml has [together] section with api_key.")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LLM using Together.ai

llm = ChatOpenAI(
    openai_api_base="https://api.together.xyz/v1",
    openai_api_key=together_api_key,
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.5,
    max_tokens=2048,
)

# Streamlit UI
st.set_page_config(page_title="🧠 RAG-CHATBOT", layout="centered")

# === Theme Selector ===
theme = st.selectbox("🎨 Choose your Theme", ["Sunset", "Ocean", "Forest", "Light"])

def apply_theme(theme):
    styles = {
        "Sunset": {"bg": "#fff3e0", "text": "#e65100", "accent": "#ff7043"},
        "Ocean": {"bg": "#e0f7fa", "text": "#01579b", "accent": "#00acc1"},
        "Forest": {"bg": "#edf7ed", "text": "#1b5e20", "accent": "#4caf50"},
        "Light": {"bg": "#f9f9f9", "text": "#000000", "accent": "#4a7c59"},
    }

    selected = styles[theme]
    st.markdown(f"""
        <style>
        html, body, [class*="css"] {{
            background-color: {selected['bg']};
            color: {selected['text']};
        }}
        .main {{
            background-color: {selected['bg']};
        }}
        h1, h2, h3, .stTextInput label {{
            color: {selected['text']};
        }}
        .stButton > button {{
            background-color: {selected['accent']};
            color: white;
        }}
        .stAlert {{
            border-left: 5px solid {selected['accent']} !important;
            background-color: rgba(0, 0, 0, 0.05);
        }}
        </style>
    """, unsafe_allow_html=True)

apply_theme(theme)

st.markdown("<h1 style='text-align: center; margin-top: 20px;'> DOCUMENTS QA-BOT</h1>", unsafe_allow_html=True)

# --- User name input ---
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.user_name:
    st.session_state.user_name = st.text_input("👤 Please Enter your Name:", max_chars=30)

# Restrict access if user_name is empty
if not st.session_state.user_name.strip():
    st.warning("⚠️ Please enter your name to continue chatting....")
    st.stop()  # stops the app here until name is provided

# Initialize session state variables
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts with query, answer, mode, timestamp, user
if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "total_questions": 0,
        "text_questions": 0,
        "voice_questions": 0,
        "questions_list": [],  # to track popular questions
    }

# Load and split document
def load_and_split(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Upload file
uploaded_file = st.file_uploader("📁 Upload PDF or DOCX file", type=["pdf", "docx"])
if uploaded_file and st.button("🔄 First Process Document"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    try:
        chunks = load_and_split(tmp_path)
        vector_store = FAISS.from_documents(chunks, embedding_model)
        retriever = vector_store.as_retriever()
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        st.success("✅ Document processed! Ask your question now.")
        os.remove(tmp_path)
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Helper function to update chat history and analytics
def update_state(query, answer, mode):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append({
        "query": query,
        "answer": answer,
        "mode": mode,
        "timestamp": timestamp,
        "user": st.session_state.user_name or "Unknown",
    })
    st.session_state.analytics["total_questions"] += 1
    st.session_state.analytics[f"{mode}_questions"] += 1
    st.session_state.analytics["questions_list"].append(query.lower())

# Text QA
st.subheader("💬 Ask with Text")
text_query = st.text_input("Type your question here:")
if st.button("Press Ask Button") and text_query:
    if st.session_state.qa_chain:
        try:
            result = st.session_state.qa_chain.invoke({"query": text_query})
            answer = result.get("result", result)
            st.write("🤖 Answer:", answer)
            update_state(text_query, answer, "text")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please upload and process a document first....")

# Voice QA
st.subheader("🎙️ Ask with Voice")
if st.button("🎤 Speak Now"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            st.info("🎙️ I am Listening... Speak now.")
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        st.success("✅ Your voice was captured. Processing...")

        query = recognizer.recognize_google(audio_data)
        st.write("🗣️ You asked:", query)

        if st.session_state.qa_chain:
            result = st.session_state.qa_chain.invoke({"query": query})
            answer = result.get("result", result)
            st.write("🤖 Answer:", answer)

            tts = gTTS(text=answer)
            tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            tts.save(tts_path)
            st.audio(tts_path, format="audio/mp3")

            update_state(query, answer, "voice")
        else:
            st.warning("⚠️ Please upload and process a document first....")

    except sr.WaitTimeoutError:
        st.error("⏰ Timeout: You didn't speak.")
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"⚠️ API error: {e}")
    except Exception as e:
        st.error(f"❌ Voice processing error: {e}")

# Download chat history with timestamps and user names
st.subheader("📥 Download Chat History")
if st.session_state.chat_history:
    chat_text = ""
    for i, entry in enumerate(st.session_state.chat_history, 1):
        chat_text += (
            f"{i}. [{entry['mode'].upper()}] {entry['timestamp']} - {entry['user']}\n"
            f"    Q: {entry['query']}\n"
            f"    A: {entry['answer']}\n\n"
        )
    st.download_button(
        label="Download chat history as TXT",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )
else:
    st.info("Chat history will appear here after you start asking questions.")

# Analytics Dashboard
st.subheader("📊 Chat Analytics Dashboard")
analytics = st.session_state.analytics

col1, col2, col3 = st.columns(3)
col1.metric("Total Questions Asked", analytics["total_questions"])
col2.metric("Text Questions", analytics["text_questions"])
col3.metric("Voice Questions", analytics["voice_questions"])

# Popular questions (top 5)
counter = Counter(analytics["questions_list"])
most_common = counter.most_common(5)
if most_common:
    st.write("Top 5 Most Asked Questions:")
    for q, count in most_common:
        st.write(f"- {q} (asked {count} times)")
else:
    st.write("No questions asked yet.")

# Recent interactions (last 5)
st.subheader("📅 Recent Interactions")
if st.session_state.chat_history:
    for entry in reversed(st.session_state.chat_history[-5:]):
        st.write(f"{entry['timestamp']} - {entry['user']} [{entry['mode'].upper()}]")
        st.write(f"Q: {entry['query']}")
        st.write(f"A: {entry['answer']}")
        st.markdown("---")
else:
    st.write("No interactions yet.")