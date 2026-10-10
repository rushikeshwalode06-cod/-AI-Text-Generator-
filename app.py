import streamlit as st
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PREMIUM DASHBOARD UI
# =========================================================
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(59,130,246,.20), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(168,85,247,.20), transparent 30%),
        linear-gradient(135deg,#070b16 0%,#111827 48%,#1e1b4b 100%);
    color:#f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Hero */
.hero {
    padding: 30px 25px;
    border-radius: 26px;
    text-align: center;
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.12);
    box-shadow: 0 20px 55px rgba(0,0,0,.25);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 48px;
    font-weight: 900;
    margin: 0;
    background: linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color:#cbd5e1;
    font-size:17px;
    margin-top:8px;
}

/* Dashboard cards */
.card {
    background: rgba(255,255,255,.065);
    border:1px solid rgba(255,255,255,.12);
    border-radius:20px;
    padding:22px;
    margin-bottom:20px;
    box-shadow:0 12px 35px rgba(0,0,0,.16);
    backdrop-filter:blur(12px);
}

.section-title {
    font-size:22px;
    font-weight:800;
    margin-bottom:10px;
}

/* Live prompt */
.live-prompt {
    background:linear-gradient(
        135deg,
        rgba(37,99,235,.20),
        rgba(59,130,246,.06)
    );
    border:1px solid rgba(96,165,250,.38);
    border-left:5px solid #60a5fa;
    border-radius:15px;
    padding:17px;
    color:#e2e8f0;
    font-size:17px;
    line-height:1.75;
    white-space:pre-wrap;
    word-wrap:break-word;
    min-height:58px;
}

.live-empty {
    color:#94a3b8;
    font-style:italic;
}

/* Output */
.output-box {
    background:linear-gradient(
        135deg,
        rgba(124,58,237,.20),
        rgba(168,85,247,.06)
    );
    border:1px solid rgba(192,132,252,.38);
    border-left:5px solid #c084fc;
    border-radius:15px;
    padding:20px;
    color:#f8fafc;
    font-size:18px;
    line-height:1.85;
    white-space:pre-wrap;
    word-wrap:break-word;
}

/* Badge */
.badge {
    display:inline-block;
    padding:6px 12px;
    border-radius:999px;
    background:rgba(96,165,250,.12);
    border:1px solid rgba(96,165,250,.25);
    color:#bfdbfe;
    font-size:12px;
    font-weight:700;
    margin-bottom:12px;
}

/* Stats */
.stat {
    background:rgba(255,255,255,.055);
    border:1px solid rgba(255,255,255,.10);
    border-radius:16px;
    padding:17px 10px;
    text-align:center;
}

.stat-value {
    font-size:25px;
    font-weight:800;
    color:#93c5fd;
}

.stat-label {
    color:#94a3b8;
    font-size:13px;
    margin-top:3px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#080d19,#111827);
    border-right:1px solid rgba(255,255,255,.08);
}

/* Text input: BLACK text so typed prompt is clearly visible */
.stTextArea textarea {
    background: #ffffff !important;
    color: #111827 !important;
    caret-color: #111827 !important;
    border: 2px solid #60a5fa !important;
    border-radius: 14px !important;
    font-size: 17px !important;
    font-weight: 500 !important;
}

.stTextArea textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* All buttons */
.stButton > button,
.stDownloadButton > button {
    width:100%;
    min-height:48px;
    border-radius:13px !important;
    font-weight:750 !important;
    font-size:15px !important;
    transition: all .2s ease !important;
}

/* Generate */
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#ef4444,#f43f5e) !important;
    color: #ffffff !important;
    border: none !important;
}

/* Normal buttons */
div.stButton > button:not([kind="primary"]) {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
}

/* Download button */
div.stDownloadButton > button {
    background: linear-gradient(90deg,#2563eb,#7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
}

/* Example prompt buttons */
.example-button {
    background:#ffffff;
    color:#111827;
    border-radius:13px;
    padding:13px;
    border:1px solid #cbd5e1;
}

/* Example prompt buttons */
div.stButton > button {
    color: #111827 !important;
    background: linear-gradient(135deg, #ffffff, #f8fafc) !important;
    border: 1px solid #cbd5e1 !important;
}

/* Hover */
.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(0,0,0,.25);
}

/* Footer */
.footer {
    text-align:center;
    color:#94a3b8;
    padding:28px 10px 5px;
}

/* Hide Streamlit default menu/header decorations */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
MODEL_NAME = "rushikeshwalode/eli5_casual_lm-model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device

# =========================================================
# SESSION STATE
# =========================================================
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = ""

if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

if "generation_time" not in st.session_state:
    st.session_state.generation_time = 0.0

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🤖 AI Text Generator</div>
    <div class="hero-subtitle">
        Generate meaningful text continuation with your trained language model
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Generation Settings")
    st.markdown("---")

    max_new_tokens = st.slider(
        "📝 Max New Tokens",
        20, 300, 100, 10
    )

    temperature = st.slider(
        "🌡️ Temperature",
        0.2, 1.2, 0.7, 0.05,
        help="Lower values make the output more focused."
    )

    top_k = st.slider(
        "🔝 Top-K",
        1, 50, 30, 1
    )

    top_p = st.slider(
        "🎯 Top-P",
        0.5, 1.0, 0.90, 0.05
    )

    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.info(
        "DistilGPT-2\n\n"
        "Fine-tuned causal language model."
    )

    if torch.cuda.is_available():
        st.success("🚀 GPU Available")
    else:
        st.warning("💻 CPU Mode")

# =========================================================
# LOAD MODEL
# =========================================================
try:
    with st.spinner("Loading your AI model..."):
        tokenizer, model, device = load_model()
except Exception as e:
    st.error("❌ Model loading failed.")
    st.code(str(e))
    st.stop()

# =========================================================
# INPUT CARD
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown(
    '<div class="badge">✍️ INPUT AREA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Write your prompt</div>',
    unsafe_allow_html=True
)

# The value is stored in session state, so the text remains visible.
prompt = st.text_area(
    "Prompt",
    value=st.session_state.prompt_text,
    height=155,
    placeholder=(
        "Start writing here...\n\n"
        "Example: The human immune system protects the body by"
    ),
    label_visibility="collapsed"
)

st.session_state.prompt_text = prompt

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# LIVE PROMPT PREVIEW
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown(
    '<div class="badge">👀 LIVE PROMPT PREVIEW</div>',
    unsafe_allow_html=True
)

if prompt.strip():
    # Show exactly what the user has typed
    st.markdown(
        f'<div class="live-prompt">{prompt}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="live-prompt live-empty">'
        'Your prompt will appear here while you type...'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ALL MAIN BUTTONS ARE ALWAYS VISIBLE
# =========================================================
b1, b2, b3 = st.columns(3)

with b1:
    generate_clicked = st.button(
        "🚀 Generate Text",
        use_container_width=True,
        type="primary"
    )

with b2:
    clear_clicked = st.button(
        "🗑️ Clear",
        use_container_width=True
    )

with b3:
    download_data = (
        st.session_state.generated_text
        if st.session_state.generated_text
        else "Please generate text first."
    )

    download_clicked = st.download_button(
        "⬇️ Download",
        data=download_data,
        file_name="generated_text.txt",
        mime="text/plain",
        use_container_width=True
    )

# =========================================================
# CLEAR
# =========================================================
if clear_clicked:
    st.session_state.prompt_text = ""
    st.session_state.generated_text = ""
    st.session_state.generation_time = 0.0
    st.rerun()

# =========================================================
# GENERATE
# =========================================================
if generate_clicked:

    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt first.")
    else:
        clean_prompt = " ".join(prompt.strip().split())

        with st.spinner("🤖 Generating meaningful text..."):
            start_time = time.time()

            inputs = tokenizer(
                clean_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,

                    # New continuation tokens
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=20,

                    # More controlled generation
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,

                    # Reduce repetition
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,

                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_text = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            ).strip()

            st.session_state.generated_text = generated_text
            st.session_state.generation_time = time.time() - start_time

# =========================================================
# GENERATED RESULT
# =========================================================
if st.session_state.generated_text:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="badge">✨ AI GENERATED RESULT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Generated Text</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="output-box">'
        f'{st.session_state.generated_text}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Stats
    input_tokens = len(
        tokenizer.encode(st.session_state.prompt_text)
    )

    output_tokens = len(
        tokenizer.encode(st.session_state.generated_text)
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="stat">'
            f'<div class="stat-value">{input_tokens}</div>'
            f'<div class="stat-label">Prompt Tokens</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="stat">'
            f'<div class="stat-value">{output_tokens}</div>'
            f'<div class="stat-label">Total Tokens</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="stat">'
            f'<div class="stat-value">'
            f'{st.session_state.generation_time:.2f}s'
            f'</div>'
            f'<div class="stat-label">Generation Time</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# =========================================================
# EXAMPLE PROMPTS
# =========================================================
st.markdown(
    '<div class="card"><div class="badge">💡 QUICK START</div>'
    '<div class="section-title">Example Prompts</div>',
    unsafe_allow_html=True
)

examples = [
    "🧬  The human immune system protects the body by",
    "🧠  Machine learning is a field of artificial intelligence that",
    "🌌  The solar system contains planets that"
]

ec1, ec2, ec3 = st.columns(3)

for index, example in enumerate(examples):
    with [ec1, ec2, ec3][index]:
        if st.button(
            f" {example}",
            use_container_width=True,
            key=f"example_{index}"
        ):
            st.session_state.prompt_text = example
            st.session_state.generated_text = ""
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer">'
    '🤖 AI Text Generator &nbsp; • &nbsp; DistilGPT-2 &nbsp; • &nbsp; Streamlit'
    '</div>',
    unsafe_allow_html=True
)
