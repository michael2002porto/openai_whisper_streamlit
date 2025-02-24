# Install dependencies

# Linux
# sudo apt update && sudo apt install ffmpeg

# MacOS
# brew install ffmpeg

# Windows
# chco install ffmpeg

# Installing pytorch
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Installing Whisper
# pip install git+https://github.com/openai/whisper.git -q

# pip install streamlit

import streamlit as st
import whisper
import tempfile
import os

st.title("Whisper ASR App")

# Upload audio file with Streamlit
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

# Load Whisper model (cache it to avoid reloading every time)
@st.cache_resource
def load_model():
    return whisper.load_model("turbo")

model = load_model()
st.text("Whisper Model Loaded")

if audio_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    st.sidebar.header("Play Original Audio File")
    st.sidebar.audio(temp_audio_path)

    # Transcribe when button is clicked
    if st.sidebar.button("Transcribe Audio"):
        st.sidebar.success("Transcribing Audio...")
        transcription = model.transcribe(temp_audio_path, language="id")
        st.sidebar.success("Transcription Complete")

        st.subheader("Transcription:")
        st.markdown(transcription["text"])

        # Clean up temp file
        os.remove(temp_audio_path)
else:
    st.sidebar.warning("Please upload an audio file first")
