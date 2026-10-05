import streamlit as st
import whisper
import json
import tempfile
import os
import srt
from datetime import timedelta
from groq import Groq

st.set_page_config(page_title="Negotiation Coach", page_icon="🎙️", layout="centered")

st.title("🎙️ Negotiation Coach")
st.caption("Transcribe audio → download captions → optionally analyze your negotiation")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Groq API Key")
    groq_key = st.text_input(
        "Groq API key (for analysis only)", type="password", placeholder="gsk_...",
        value=st.secrets.get("GROQ_API_KEY", "")
    )
    st.caption("Free at console.groq.com — no credit card needed.")

    st.divider()
    st.header("⚙️ Whisper settings")
    model_size = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium", "large"],
        index=2,
        help="small = good balance of speed & accuracy"
    )
    quality = st.select_slider(
        "Transcription quality",
        options=["Light", "Medium", "High"],
        value="Medium",
        help="Light=tiny model, Medium=small, High=large"
    )
    # quality overrides model size
    quality_model = {"Light": "tiny", "Medium": "small", "High": "large"}[quality]

    output_format = st.radio(
        "Caption format",
        ["TXT", "SRT"],
        help="TXT = plain text, SRT = subtitles with timestamps"
    )

# ── Cache Whisper model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Whisper model…")
def load_whisper_model(size):
    return whisper.load_model(size)

# ── Helpers ────────────────────────────────────────────────────────────────────
def segments_to_srt(segments):
    subs = []
    for i, seg in enumerate(segments, start=1):
        start = timedelta(seconds=seg["start"])
        end = timedelta(seconds=seg["end"])
        subs.append(srt.Subtitle(index=i, start=start, end=end, content=seg["text"].strip()))
    return srt.compose(subs)

def segments_to_txt(segments):
    return "\n".join(seg["text"].strip() for seg in segments)

def flag(lang_code):
    flags = {"en":"🇬🇧","fr":"🇫🇷","es":"🇪🇸","de":"🇩🇪","pt":"🇵🇹","it":"🇮🇹","nl":"🇳🇱","zh":"🇨🇳","ja":"🇯🇵","ko":"🇰🇷","ar":"🇸🇦","ru":"🇷🇺"}
    return flags.get(lang_code, "🌐")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload & Transcribe
# ══════════════════════════════════════════════════════════════════════════════
st.header("Step 1 — Upload & transcribe")

audio_file = st.file_uploader(
    "Drop your audio file here",
    type=["mp3", "mp4", "wav", "m4a", "webm", "ogg", "flac"],
)

if audio_file:
    st.audio(audio_file)
    st.caption(f"📎 {audio_file.name} — {audio_file.size / 1024 / 1024:.1f} MB")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🎚️ Quality: **{quality}** (model: `{quality_model}`)")
    with col2:
        st.info(f"📄 Format: **{output_format}**")

    if st.button("🎧 Transcribe", type="primary", use_container_width=True):
        with st.spinner(f"Transcribing with Whisper ({quality_model})… this may take a moment."):
            try:
                suffix = os.path.splitext(audio_file.name)[1] or ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                model = load_whisper_model(quality_model)
                # Auto-detect language (no language param = auto)
                result = model.transcribe(tmp_path, word_timestamps=False)
                os.unlink(tmp_path)

                detected_lang = result.get("language", "unknown")
                segments = result["segments"]
                full_text = result["text"].strip()

                st.session_state["segments"] = segments
                st.session_state["full_text"] = full_text
                st.session_state["detected_lang"] = detected_lang
                st.session_state["output_format"] = output_format
                st.session_state["analysis_done"] = False

                st.success(f"✅ Done — {flag(detected_lang)} Language detected: **{detected_lang.upper()}** — {len(full_text.split())} words")

            except Exception as e:
                st.error(f"Transcription error: {e}")

# ── Show transcript + download ─────────────────────────────────────────────────
if st.session_state.get("segments"):
    segments = st.session_state["segments"]
    full_text = st.session_state["full_text"]
    detected_lang = st.session_state["detected_lang"]
    fmt = st.session_state.get("output_format", output_format)

    st.divider()
    st.subheader(f"📄 Transcript — {flag(detected_lang)} {detected_lang.upper()}")

    # Preview
    with st.expander("Preview transcript", expanded=True):
        st.text_area("", value=full_text, height=200, label_visibility="collapsed")

    # Generate caption file
    if fmt == "SRT":
        caption_content = segments_to_srt(segments)
        filename = "captions.srt"
        mime = "text/plain"
    else:
        caption_content = segments_to_txt(segments)
        filename = "transcript.txt"
        mime = "text/plain"

    st.download_button(
        label=f"⬇️ Download {fmt} file",
        data=caption_content,
        file_name=filename,
        mime=mime,
        type="primary",
        use_container_width=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Optional: Negotiation Analysis
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.header("Step 2 — Analyze your negotiation (optional)")
    st.caption("Identify your role, detect biases, and get coaching on how to improve.")

    col_a, col_b = st.columns(2)
    with col_a:
        ROLES = ["Candidate", "Hiring manager", "Buyer", "Seller", "Employee", "Manager", "Mediator", "Other…"]
        role_choice = st.selectbox("Your role in this conversation", ROLES)
        if role_choice == "Other…":
            role = st.text_input("Describe your role", placeholder="e.g. Startup founder")
        else:
            role = role_choice

    with col_b:
        st.markdown("**How should Groq recognize you?**")
        speaker_hint = st.text_area(
            "Paste a few sentences you said, or describe how you speak",
            placeholder="e.g. I'm the one asking about salary, saying things like 'I was thinking more around…'",
            height=120,
            label_visibility="collapsed"
        )

    if st.button("🔍 Analyze negotiation", type="primary", use_container_width=True):
        if not groq_key:
            st.error("Add your Groq API key in the sidebar.")
        elif not role.strip():
            st.error("Please specify your role.")
        else:
            prompt = f"""You are an expert negotiation coach and behavioral analyst.

The user's role is: "{role}"
To identify the user in the transcript, use this hint: "{speaker_hint if speaker_hint else 'infer from role'}"

Transcript:
\"\"\"
{full_text}
\"\"\"

Return ONLY valid JSON (no markdown fences) with this structure:
{{
  "speaker_identified": "how you identified the user's voice/turns in the transcript",
  "summary": "2-sentence summary of the negotiation",
  "scores": {{
    "assertiveness": <0-100>,
    "clarity": <0-100>,
    "empathy": <0-100>,
    "leverage": <0-100>,
    "preparation": <0-100>
  }},
  "insights": [
    {{"type": "strength", "label": "Strength identified", "text": "..."}},
    {{"type": "bias", "label": "Cognitive bias detected", "text": "..."}},
    {{"type": "tactic", "label": "Recommended tactic", "text": "..."}},
    {{"type": "risk", "label": "Risk to watch", "text": "..."}}
  ],
  "power_dynamic": "who holds leverage and why",
  "next_move": "specific actionable recommendation for {role} in 2-3 sentences",
  "reframe": "an alternative phrase or framing they could use"
}}

Be specific. Detect real cognitive biases (anchoring, loss aversion, BATNA blindness, reactive devaluation, status quo bias, etc.)."""

            with st.spinner("Groq is analyzing…"):
                try:
                    client = Groq(api_key=groq_key)
                    response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024,
                        temperature=0.3,
                    )
                    raw = response.choices[0].message.content.strip()
                    data = json.loads(raw.replace("```json", "").replace("```", "").strip())
                    st.session_state["analysis"] = data
                    st.session_state["analysis_role"] = role

                except json.JSONDecodeError:
                    st.error("Could not parse response. Try again.")
                except Exception as e:
                    st.error(f"Analysis error: {e}")

    # ── Show analysis results ──────────────────────────────────────────────────
    if st.session_state.get("analysis"):
        data = st.session_state["analysis"]
        role = st.session_state["analysis_role"]

        st.divider()
        st.subheader("📊 Negotiation analysis")

        if data.get("speaker_identified"):
            st.caption(f"🎯 Speaker identified as: {data['speaker_identified']}")

        st.markdown(f"**Situation ({role}):** {data['summary']}")
        st.info(f"⚖️ **Power dynamic:** {data['power_dynamic']}")

        st.markdown("#### Scores")
        score_labels = ["assertiveness", "clarity", "empathy", "leverage", "preparation"]
        cols = st.columns(5)
        for col, key in zip(cols, score_labels):
            val = data["scores"][key]
            icon = "🟢" if val >= 70 else "🟡" if val >= 45 else "🔴"
            col.metric(key.capitalize(), f"{icon} {val}")
        for key in score_labels:
            val = data["scores"][key]
            st.progress(val / 100, text=f"{key.capitalize()}: {val}/100")

        st.markdown("#### Insights")
        icons = {"strength": "✅", "bias": "⚠️", "tactic": "💡", "risk": "🚨"}
        for ins in data.get("insights", []):
            with st.expander(f"{icons.get(ins['type'], '•')} {ins['label']}", expanded=True):
                st.write(ins["text"])

        st.markdown("#### Your next move")
        st.success(data["next_move"])
        if data.get("reframe"):
            st.markdown(f"💬 **Try reframing it as:** *\"{data['reframe']}\"*")
