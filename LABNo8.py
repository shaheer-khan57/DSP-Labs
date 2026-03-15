# LABNo8.py – FINAL 100% WORKING VERSION (Clean + Noisy Spectrograms)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO

st.set_page_config(page_title="DSP Lab 8 – Complete Analysis", layout="wide")
st.title("DSP Lab 8 – Audio + Noise + Full Separate Analysis")
st.markdown("**Clean & Noisy: Waveform │ FFT │ Spectrogram – All side by side**")

# -------------------------- Sidebar --------------------------
st.sidebar.header("Noise Control")
noise_level = st.sidebar.slider("Noise amplitude", 0.001, 0.20, 0.03, 0.001)

# -------------------------- Record Audio --------------------------
st.subheader("Step 1: Record Your Voice (5–10 seconds)")
audio_file = st.audio_input("Click mic → speak → stop", key="recorder")

if audio_file is not None:
    audio_bytes = audio_file.read()
    audio_np, fs = sf.read(BytesIO(audio_bytes))

    # Stereo → Mono
    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=1)

    # Normalize
    audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-9)

    st.success(f"Recorded {len(audio_np)/fs:.2f} s @ {fs:,} Hz")
    st.audio(audio_bytes, format="audio/wav")

    if st.button("Add Noise & Analyze Everything", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            noise = np.random.normal(0, noise_level, len(audio_np))
            noisy_np = audio_np + noise
            noisy_np = noisy_np / (np.max(np.abs(noisy_np)) + 1e-9)

            buffer = BytesIO()
            sf.write(buffer, noisy_np, fs, format="WAV")
            noisy_bytes = buffer.getvalue()

            st.session_state.update({
                "clean_np": audio_np,
                "noisy_np": noisy_np,
                "fs": fs,
                "clean_bytes": audio_bytes,
                "noisy_bytes": noisy_bytes
            })
        st.success("Analysis complete!")
        st.rerun()

# -------------------------- Results --------------------------
if "noisy_np" in st.session_state:
    clean_np    = st.session_state.clean_np
    noisy_np    = st.session_state.noisy_np
    fs          = st.session_state.fs
    clean_bytes = st.session_state.clean_bytes
    noisy_bytes = st.session_state.noisy_bytes

    time = np.linspace(0, len(clean_np)/fs, len(clean_np))
    freqs = np.fft.rfftfreq(len(clean_np), 1/fs)

    # Audio players
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.subheader("Clean Audio")
        st.audio(clean_bytes, format="audio/wav")
    with col2:
        st.subheader("Noisy Audio")
        st.audio(noisy_bytes, format="audio/wav")
    with col3:
        st.download_button("Download Noisy.wav", noisy_bytes, "noisy_audio.wav", "audio/wav", use_container_width=True)

    st.markdown("---")

    # Time domain
    st.subheader("Time Domain Waveforms")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(time, clean_np, color="#1f77b4")
        ax.set_title("Clean Signal", fontweight="bold")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(time, noisy_np, color="#d62728")
        ax.set_title("Noisy Signal", fontweight="bold")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # FFT
    st.subheader("Magnitude Spectrum (FFT)")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.semilogy(freqs, np.abs(np.fft.rfft(clean_np)), color="#1f77b4")
        ax.set_title("Clean FFT", fontweight="bold")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude")
        ax.set_xlim(0, 8000); ax.grid(alpha=0.3)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.semilogy(freqs, np.abs(np.fft.rfft(noisy_np)), color="#d62728")
        ax.set_title("Noisy FFT", fontweight="bold")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude")
        ax.set_xlim(0, 8000); ax.grid(alpha=0.3)
        st.pyplot(fig)

    # CLEAN & NOISY SPECTROGRAMS (fixed & optimized)
    st.subheader("Spectrograms – Clean vs Noisy")
    c1, c2 = st.columns(2)

    with c1:
        fig = plt.figure(figsize=(11, 5))
        ax = fig.add_subplot(111)
        Pxx, freqs_s, t_s, im = ax.specgram(clean_np, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
        ax.set_ylim(0, 8000)
        ax.set_title("Spectrogram – Clean Signal", fontweight="bold", fontsize=14)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax, label="Intensity (dB)")
        st.pyplot(fig)

    with c2:
        fig = plt.figure(figsize=(11, 5))
        ax = fig.add_subplot(111)
        Pxx, freqs_s, t_s, im = ax.specgram(noisy_np, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
        ax.set_ylim(0, 8000)
        ax.set_title("Spectrogram – Noisy Signal", fontweight="bold", fontsize=14)
        ax.set_xlabel("Time (s)s"); ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax, label="Intensity (dB)")
        st.pyplot(fig)

    # Zoomed view
    st.subheader("Zoomed Waveform – First 0.5 s")
    zoom = int(0.5 * fs)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.plot(clean_np[:zoom], color="#1f77b4")
        ax.set_title("Clean (0–0.5 s)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.plot(noisy_np[:zoom], color="#d62728", alpha=0.9)
        ax.set_title("Noisy (0–0.5 s)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # Clear button
    if st.button("Clear & Record Again", type="secondary", use_container_width=True):
        for key in ["clean_np","noisy_np","fs","clean_bytes","noisy_bytes"]:
            st.session_state.pop(key, None)
        st.rerun()

else:
    st.info("Record your voice → click **Add Noise & Analyze Everything** → see beautiful side-by-side results!")
    st.balloons()

st.caption("DSP Lab 8 – Final Complete Version | 2025")