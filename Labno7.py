
import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from io import BytesIO

# Page settings
st.set_page_config(page_title="DSP Speech Recording App", layout="wide")
st.title("Lab No. 7: Speech Recording & Visualization Using Streamlit")

# Helper functions
def ensure_folder(path="recordings"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def plot_waveform(signal, title="Speech Signal"):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(signal)
    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

def audio_bytes_from_array(arr, fs):
    buf = BytesIO()
    sf.write(buf, arr, fs, format="WAV")
    buf.seek(0)
    return buf.read()

# UI Controls
col1, col2 = st.columns([2, 1])

with col1:
    duration = st.slider("Recording Duration (seconds)", 1, 10, 3)
    fs = st.selectbox("Sampling Rate (Hz)", [16000, 22050, 44100], index=2)

with col2:
    save_dir = ensure_folder("recordings")
    st.write("Recordings Folder:")
    st.write(save_dir)

# Record button
if st.button("Record Speech"):
    try:
        st.info("Recording... Please speak now.")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        ts = int(time.time())
        file_path = os.path.join(save_dir, f"speech_{ts}.wav")
        sf.write(file_path, audio, fs)

        st.success("Recording completed successfully!")

        # Play audio
        st.audio(audio_bytes_from_array(audio, fs), format="audio/wav")

        # Plot waveform
        st.pyplot(plot_waveform(audio, title="Recorded Speech Signal"))

        # Download button
        with open(file_path, "rb") as f:
            st.download_button(
                label="Download Recorded Audio",
                data=f,
                file_name=os.path.basename(file_path),
                mime="audio/wav"
            )

        st.write("Saved File Path:")
        st.write(file_path)

    except Exception as e:
        st.error(f"Recording failed: {e}")

st.markdown("---")
st.write(
    "**Note:** This lab demonstrates speech signal acquisition, visualization, "
    "and a basic user interface using Streamlit."
)
