# LABNo10_Voice_Analysis.py – Upload Your Own Voice + Spectral Analysis (2025)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf
from io import BytesIO
import tempfile

st.set_page_config(page_title="DSP Lab 10 – Your Voice Spectrum", layout="wide")
st.title("DSP Lab 10 – Upload Your Voice & See Its Spectrum")
st.markdown("**Upload a short voice recording (.wav or .mp3) → See waveform, FFT, dominant frequency & spectrogram**")

# -------------------------- Upload Audio --------------------------
uploaded_file = st.file_uploader("Upload your voice recording (WAV or MP3 recommended, 5–15 seconds)", 
                                 type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    # Read audio bytes
    audio_bytes = uploaded_file.read()
    
    # Load audio using soundfile (supports more formats)
    try:
        audio_np, fs = sf.read(BytesIO(audio_bytes))
    except:
        st.error("Could not read audio file. Try WAV format.")
        st.stop()
    
    # Convert to mono if stereo
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    
    # Normalize
    audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-9)
    
    duration = len(audio_np) / fs
    t = np.linspace(0, duration, len(audio_np))

    st.audio(audio_bytes, format="audio/wav")
    st.success(f"Loaded: {duration:.2f} seconds | Sampling Rate: {fs:,} Hz")

    # -------------------------- Compute FFT --------------------------
    def compute_fft(signal, fs):
        N = len(signal)
        freqs = np.fft.rfftfreq(N, 1/fs)
        mag = np.abs(np.fft.rfft(signal)) / N
        mag[1:-1] *= 2
        return freqs, mag

    freqs, mag = compute_fft(audio_np, fs)

    # Find dominant frequency
    peak_idx = np.argmax(mag[1:]) + 1  # skip DC
    dominant_freq = freqs[peak_idx]
    peak_mag = mag[peak_idx]

    # -------------------------- Plots --------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Time Domain Waveform (Your Voice)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, audio_np, color="steelblue", linewidth=1)
        ax.set_title("Waveform of Uploaded Voice")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.subheader("Spectrogram (Frequency over Time)")
        fig, ax = plt.subplots(figsize=(10, 5))
        Pxx, freq_spec, t_spec, im = ax.specgram(audio_np, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
        ax.set_ylim(0, 5000)  # Focus on voice range
        ax.set_title("Spectrogram – Shows Formants in Speech")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax, label="Intensity (dB)")
        st.pyplot(fig)

    with col2:
        st.subheader("Magnitude Spectrum (FFT)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freqs, mag, color="crimson", linewidth=1.5)
        ax.set_title("Frequency Content of Your Voice")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(0, 5000)  # Human voice mostly below 5kHz
        ax.set_ylim(0, peak_mag * 1.2)
        ax.grid(alpha=0.3)
        
        # Highlight dominant frequency
        ax.axvline(dominant_freq, color="green", linestyle="--", linewidth=2)
        ax.annotate(f"Dominant Frequency\n{dominant_freq:.0f} Hz", 
                    xy=(dominant_freq, peak_mag), xytext=(dominant_freq + 300, peak_mag * 0.8),
                    arrowprops=dict(arrowstyle="->", color="green", lw=2),
                    color="green", fontsize=14, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        st.pyplot(fig)

    # -------------------------- Observations --------------------------
    st.markdown("---")
    st.subheader("Key Observations from Your Voice")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duration", f"{duration:.2f} s")
    with col2:
        st.metric("Dominant Frequency", f"{dominant_freq:.0f} Hz")
        st.write("Usually 85–180 Hz (male), 165–255 Hz (female)")
    with col3:
        st.metric("Voice Type Hint", 
                  "Male-like" if dominant_freq < 150 else "Female-like" if dominant_freq < 250 else "High-pitched")

    st.info("""
    **What you see:**
    - **Waveform**: Complex, not a pure sine → real speech is sum of many frequencies
    - **FFT**: Multiple peaks → formants (resonances of vocal tract)
    - **Spectrogram**: Dark horizontal bands → formants changing as you speak
    - **Dominant peak** → Your fundamental pitch (F0)
    """)

    # Download processed data
    st.markdown("---")
    output = BytesIO()
    sf.write(output, audio_np, fs, format="WAV")
    st.download_button("Download Processed Audio (Normalized)", 
                       data=output.getvalue(), 
                       file_name="my_voice_normalized.wav", 
                       mime="audio/wav")

else:
    st.info("👆 Upload a short voice recording (say 'Hello, this is my voice' clearly)")
    st.markdown("""
    **Tips for best results:**
    - Record in a quiet room
    - Speak clearly for 5–10 seconds
    - Use .wav format if possible
    - Try saying vowels (Aaa, Eee, Ooo) to see formants clearly!
    """)
    st.balloons()

st.caption("DSP Lab 10 – Analyze Your Own Voice | Waveform + FFT + Spectrogram | 2025")