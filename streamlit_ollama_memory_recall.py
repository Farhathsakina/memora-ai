import os
import json
import requests
import streamlit as st

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "mistral"
MEMORY_FILE = "memory.json"
MAX_CONTEXT_MESSAGES = 6

st.set_page_config(page_title="Ollama Memory Recall Chat", layout="wide")

st.markdown("""
<style>

/* ===== FULL BLUE BACKGROUND FIX ===== */
html, body, 
[data-testid="stAppViewContainer"], 
[data-testid="stHeader"], 
[data-testid="stToolbar"] {
    background: linear-gradient(135deg,#0f172a,#020617) !important;
    color: white;
}

/* Remove default header background */
header {
    background: transparent !important;
}

/* Reduce top spacing */
.block-container {
    padding-top: 1rem;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #0f172a;
}

/* Chat styling */
.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

/* Rounded input */
.stTextInput > div > div > input {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"users": {}}

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4)

def get_user(memory, user_id):
    if user_id not in memory["users"]:
        memory["users"][user_id] = {
            "name": "",
            "history": [],
            "important": []
        }
    return memory["users"][user_id]

def add_history(memory, user_id, user_text, ai_text):
    history = memory["users"][user_id]["history"]
    history.append({"user": user_text, "ai": ai_text})

    if len(history) > MAX_CONTEXT_MESSAGES:
        memory["users"][user_id]["history"] = history[-MAX_CONTEXT_MESSAGES:]

    save_memory(memory)

def build_prompt(user, user_input):
    return f"""
You are a helpful AI assistant.
Answer clearly and simply.

User: {user_input}
Assistant:
""".strip()

def ollama_stream(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=None
    )

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            yield data.get("response", "")
            if data.get("done"):
                break

st.markdown("""
<div style='text-align:center; padding:20px 0;'>
<h1 style='font-size:50px; font-weight:800;
background: linear-gradient(90deg,#38bdf8,#a78bfa);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;'>
🧠 Ollama AI Memory Assistant
</h1>
<p style='color:gray; font-size:18px;'>
Offline • Persistent Memory • Smart Recall
</p>
</div>
""", unsafe_allow_html=True)

memory = load_memory()

st.sidebar.markdown("## 👤 User Profile")

user_id = st.sidebar.text_input("User ID", "user1")
user = get_user(memory, user_id)

name = st.sidebar.text_input("Your Name", value=user.get("name", ""))

if name:
    user["name"] = name
    save_memory(memory)

if st.sidebar.button("🗑️ Clear My Memory"):
    user["history"] = []
    save_memory(memory)
    st.sidebar.success("Memory cleared!")

st.sidebar.markdown("---")

st.sidebar.markdown(f"""
<div style='background:#1e293b; padding:15px; border-radius:15px;'>
<p><b>🤖 Model:</b> {MODEL_NAME}</p>
<p><b>💾 Stored Memories:</b> {len(user["history"])}</p>
<p><b>🆔 Active ID:</b> {user_id}</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.metric("💬 Conversations", len(user["history"]))
col2.metric("🤖 Model", MODEL_NAME)
col3.metric("👤 User", name if name else "Guest")

st.markdown(f"""
<div style='background:#0f172a;
padding:20px;
border-radius:20px;
border:1px solid #1e293b;
margin:20px 0;'>

<h3>👋 Welcome {name or 'User'}</h3>
<p style='color:gray;'>
Your assistant remembers previous conversations and responds intelligently.
</p>

</div>
""", unsafe_allow_html=True)

for msg in user["history"]:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["ai"])

user_input = st.chat_input("Ask anything (including about past chats)...")

if user_input:
    st.chat_message("user").write(user_input)

    prompt = build_prompt(user, user_input)

    with st.chat_message("assistant"):
        response_text = ""
        placeholder = st.empty()

        for token in ollama_stream(prompt):
            response_text += token
            placeholder.markdown(response_text)

    add_history(memory, user_id, user_input, response_text)