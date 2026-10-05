import streamlit as st
import whisper
import json
import tempfile
import os
import srt
import threading
import time
from datetime import timedelta
from groq import Groq

st.set_page_config(page_title="Negotiation Coach", page_icon="🎙️", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectSlider label,
[data-testid="stSidebar"] .stRadio label { color: #aaa !important; }

/* Main text */
h1, h2, h3, h4 { color: #ffffff !important; }
p, label, .stMarkdown { color: #d0d0d0 !important; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(102,126,234,0.4);
}
.hero h1 { font-size: 3rem !important; margin: 0 !important; color: white !important; }
.hero p { font-size: 1.1rem; color: rgba(255,255,255,0.85) !important; margin-top: 0.5rem; }

/* Step cards */
.step-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.step-number {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-weight: 700;
    font-size: 0.8rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin-bottom: 0.75rem;
    letter-spacing: 0.05em;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(102,126,234,0.6) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #11998e, #38ef7d) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 20px rgba(17,153,142,0.4) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05) !important;
    border: 2px dashed rgba(102,126,234,0.5) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: white !important; font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { color: #aaa !important; }

/* Progress bar */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
}

/* Info / success / error boxes */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* Text area */
textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* Select / radio */
[data-testid="stSelectbox"] > div,
[data-testid="stRadio"] > div { color: #e0e0e0 !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* Caption */
.stCaption { color: #888 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        groq_key = st.text_input("API key", type="password")

    st.markdown("---")
    quality = st.select_slider("Transcription quality", options=["Light", "Medium", "High"], value="Medium")
    quality_model = {"Light": "tiny", "Medium": "small", "High": "large"}[quality]
    output_format = st.radio("Caption format", ["TXT", "SRT"])

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(size):
    return whisper.load_model(size)

def segments_to_srt(segments):
    subs = [srt.Subtitle(index=i+1,
        start=timedelta(seconds=s["start"]),
        end=timedelta(seconds=s["end"]),
        content=s["text"].strip()) for i, s in enumerate(segments)]
    return srt.compose(subs)

def segments_to_txt(segments):
    return "\n".join(s["text"].strip() for s in segments)

def flag(lang_code):
    flags = {"en":"🇬🇧","fr":"🇫🇷","es":"🇪🇸","de":"🇩🇪","pt":"🇵🇹","it":"🇮🇹","nl":"🇳🇱","zh":"🇨🇳","ja":"🇯🇵","ko":"🇰🇷","ar":"🇸🇦","ru":"🇷🇺"}
    return flags.get(lang_code, "🌐")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎙️ Negotiation Coach</h1>
  <p>Upload your audio · Get captions · Unlock negotiation insights</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-card">', unsafe_allow_html=True)
st.markdown('<span class="step-number">STEP 1</span>', unsafe_allow_html=True)
st.markdown("### 📂 Upload audio")

audio_file = st.file_uploader("Drop your file here — MP3, MP4, WAV, M4A, WEBM, OGG, FLAC",
    type=["mp3","mp4","wav","m4a","webm","ogg","flac"])

if audio_file:
    st.audio(audio_file)
    col1, col2, col3 = st.columns(3)
    col1.metric("File", audio_file.name[:20])
    col2.metric("Size", f"{audio_file.size/1024/1024:.1f} MB")
    col3.metric("Quality", quality)

    if st.button("▶ Transcribe", use_container_width=True):
        duration_min = audio_file.size / (1024*1024) / 1.5
        speed = {"tiny": 0.5, "small": 1.5, "large": 4.0}[quality_model]
        est_seconds = max(10, int(duration_min * speed * 60))
        est_str = f"{est_seconds//60}m {est_seconds%60}s" if est_seconds >= 60 else f"~{est_seconds}s"

        st.markdown(f"⏱️ **Estimated time:** `{est_str}`")
        bar = st.progress(0, text="Starting…")
        stop_flag = threading.Event()

        def animate():
            for i in range(1, 91):
                if stop_flag.is_set(): break
                remaining = max(0, est_seconds - int((i/90)*est_seconds))
                rem_str = f"{remaining//60}m {remaining%60}s" if remaining >= 60 else f"{remaining}s"
                bar.progress(i/100, text=f"Processing… {rem_str} remaining")
                time.sleep(est_seconds/90)

        threading.Thread(target=animate, daemon=True).start()

        try:
            suffix = os.path.splitext(audio_file.name)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            model = load_model(quality_model)
            result = model.transcribe(tmp_path, word_timestamps=False)
            os.unlink(tmp_path)
            stop_flag.set()
            bar.progress(1.0, text="✅ Complete!")

            detected_lang = result.get("language", "unknown")
            st.session_state.update({
                "segments": result["segments"],
                "full_text": result["text"].strip(),
                "detected_lang": detected_lang,
                "output_format": output_format,
                "analysis": None,
            })
            st.success(f"✅ {flag(detected_lang)} Language detected: **{detected_lang.upper()}** · {len(result['text'].split())} words")

        except Exception as e:
            stop_flag.set()
            bar.empty()
            st.error(f"Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Download captions
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("segments"):
    segments = st.session_state["segments"]
    full_text = st.session_state["full_text"]
    detected_lang = st.session_state["detected_lang"]
    fmt = st.session_state.get("output_format", output_format)

    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">STEP 2</span>', unsafe_allow_html=True)
    st.markdown(f"### 📄 Transcript — {flag(detected_lang)} {detected_lang.upper()}")

    with st.expander("Preview transcript", expanded=True):
        st.text_area("", value=full_text, height=220, label_visibility="collapsed")

    caption_content = segments_to_srt(segments) if fmt == "SRT" else segments_to_txt(segments)
    filename = "captions.srt" if fmt == "SRT" else "transcript.txt"

    st.download_button(
        label=f"⬇️ Download {fmt} file",
        data=caption_content,
        file_name=filename,
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Analysis
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">STEP 3 — OPTIONAL</span>', unsafe_allow_html=True)
    st.markdown("### 🧠 Analyze your negotiation")
    st.caption("Identify biases, power dynamics, and get a personalized coaching report.")

    col_a, col_b = st.columns(2)
    with col_a:
        ROLES = ["Candidate","Hiring manager","Buyer","Seller","Employee","Manager","Mediator","Other…"]
        role_choice = st.selectbox("Your role", ROLES)
        role = st.text_input("Describe your role", placeholder="e.g. Startup founder") if role_choice == "Other…" else role_choice
    with col_b:
        speaker_hint = st.text_area("How to recognize you in the conversation",
            placeholder="e.g. I'm the one asking about salary…", height=120)

    if st.button("🔍 Analyze negotiation", use_container_width=True):
        if not groq_key:
            st.error("API key missing — contact the administrator.")
        elif not role.strip():
            st.error("Please specify your role.")
        else:
            prompt = f"""You are an expert negotiation coach and behavioral analyst.
The user's role is: "{role}"
To identify the user: "{speaker_hint if speaker_hint else 'infer from role'}"
Transcript: \"\"\"{full_text}\"\"\"
Return ONLY valid JSON:
{{
  "speaker_identified": "...",
  "summary": "2-sentence summary",
  "scores": {{"assertiveness":<0-100>,"clarity":<0-100>,"empathy":<0-100>,"leverage":<0-100>,"preparation":<0-100>}},
  "insights": [
    {{"type":"strength","label":"...","text":"..."}},
    {{"type":"bias","label":"...","text":"..."}},
    {{"type":"tactic","label":"...","text":"..."}},
    {{"type":"risk","label":"...","text":"..."}}
  ],
  "power_dynamic":"...",
  "next_move":"...",
  "reframe":"..."
}}
Detect real biases: anchoring, loss aversion, BATNA blindness, reactive devaluation, status quo bias."""

            with st.spinner("Analyzing…"):
                try:
                    client = Groq(api_key=groq_key)
                    response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=1024, temperature=0.3,
                    )
                    raw = response.choices[0].message.content.strip()
                    data = json.loads(raw.replace("```json","").replace("```","").strip())
                    st.session_state["analysis"] = data
                    st.session_state["analysis_role"] = role
                except json.JSONDecodeError:
                    st.error("Could not parse response. Try again.")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.get("analysis"):
        data = st.session_state["analysis"]
        role = st.session_state["analysis_role"]

        st.markdown("---")
        st.markdown("### 📊 Your report")

        if data.get("speaker_identified"):
            st.caption(f"🎯 {data['speaker_identified']}")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Situation:** {data['summary']}")
        with col2:
            st.warning(f"**⚖️ Power dynamic:** {data['power_dynamic']}")

        st.markdown("#### Scores")
        score_labels = ["assertiveness","clarity","empathy","leverage","preparation"]
        cols = st.columns(5)
        for col, key in zip(cols, score_labels):
            val = data["scores"][key]
            icon = "🟢" if val >= 70 else "🟡" if val >= 45 else "🔴"
            col.metric(key.capitalize(), f"{icon} {val}")
        for key in score_labels:
            st.progress(data["scores"][key]/100, text=f"{key.capitalize()}: {data['scores'][key]}/100")

        st.markdown("#### Insights")
        icons = {"strength":"✅","bias":"⚠️","tactic":"💡","risk":"🚨"}
        for ins in data.get("insights",[]):
            with st.expander(f"{icons.get(ins['type'],'•')} {ins['label']}", expanded=True):
                st.write(ins["text"])

        st.markdown("#### Your next move")
        st.success(data["next_move"])
        if data.get("reframe"):
            st.markdown(f"💬 **Try reframing it as:** *\"{data['reframe']}\"*")

    st.markdown('</div>', unsafe_allow_html=True)
