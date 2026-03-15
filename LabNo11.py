# LABNo11_Noise_Addition_Removal_HighFreq_With_Voice.py – Voice Option Added (2025)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import soundfile as sf
from io import BytesIO

st.set_page_config(page_title="DSP Lab 11 – Noise Addition & Removal", layout="wide")
st.title("DSP Lab 11 – Noise Addition and Removal from a Higher Frequency Cumulative Signal")
st.markdown("**Record/Upload voice or generate synthetic signal → Add noise → Denoise → Compare**")

# -------------------------- Input Selection --------------------------
input_mode = st.sidebar.radio("Input Signal Mode", ["Record/Upload Voice", "Synthetic High-Freq Signal"])

if input_mode == "Record/Upload Voice":
    st.subheader("Step 1: Provide Your Voice Signal")
    col1, col2 = st.columns(2)
    with col1:
        audio_file = st.audio_input("Record your voice (5–15 seconds)", key="voice_rec")
    with col2:
        uploaded_file = st.file_uploader("Or upload audio (WAV/MP3)", type=["wav", "mp3"])

    if audio_file is not None:
        audio_bytes = audio_file.read()
        signal_clean, fs = sf.read(BytesIO(audio_bytes))
        if signal_clean.ndim > 1:
            signal_clean = np.mean(signal_clean, axis=1)
        signal_clean = signal_clean / (np.max(np.abs(signal_clean)) + 1e-9)
        t = np.linspace(0, len(signal_clean)/fs, len(signal_clean))
        st.success(f"Voice recorded: {len(signal_clean)/fs:.2f} seconds @ {fs:,} Hz")
        st.audio(audio_bytes)

    elif uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        signal_clean, fs = sf.read(BytesIO(audio_bytes))
        if signal_clean.ndim > 1:
            signal_clean = np.mean(signal_clean, axis=1)
        signal_clean = signal_clean / (np.max(np.abs(signal_clean)) + 1e-9)
        t = np.linspace(0, len(signal_clean)/fs, len(signal_clean))
        st.success(f"Voice uploaded: {len(signal_clean)/fs:.2f} seconds @ {fs:,} Hz")
        st.audio(audio_bytes)

    else:
        st.info("Record or upload your voice to proceed")
        st.stop()

else:
    # Synthetic mode (original)
    st.sidebar.header("Synthetic Signal Parameters")
    duration = st.sidebar.slider("Duration (s)", 0.5, 5.0, 2.0)
    fs = st.sidebar.slider("Sampling Rate (Hz)", 1000, 10000, 4000)
    num_freqs = st.sidebar.slider("Number of Frequencies", 1, 5, 3)
    frequencies = [st.sidebar.slider(f"Freq {i+1} (Hz)", 500, 3000, 800 + i*400) for i in range(num_freqs)]
    amplitudes = [st.sidebar.slider(f"Amp {i+1}", 0.2, 1.0, 0.6 - i*0.1) for i in range(num_freqs)]

    t = np.linspace(0, duration, int(fs * duration))
    signal_clean = np.zeros_like(t)
    for f, a in zip(frequencies, amplitudes):
        signal_clean += a * np.sin(2 * np.pi * f * t)
    signal_clean /= np.max(np.abs(signal_clean))

    st.success("Synthetic high-frequency signal generated")

# -------------------------- Add Noise (common for both modes) --------------------------
noise_level = st.sidebar.slider("Gaussian Noise Level", 0.0, 0.5, 0.15, 0.01)

if st.button("Add Noise & Denoise", type="primary"):
    noise = np.random.normal(0, noise_level, len(signal_clean))
    signal_noisy = signal_clean + noise
    signal_noisy /= np.max(np.abs(signal_noisy))

    # -------------------------- Denoising Options --------------------------
    filter_type = st.sidebar.selectbox("Denoising Filter", ["Moving Average", "Butterworth Low-Pass"])

    if filter_type == "Moving Average":
        ma_length = st.sidebar.slider("Moving Average Length", 3, 51, 11)
        kernel = np.ones(ma_length) / ma_length
        signal_denoised = np.convolve(signal_noisy, kernel, mode='same')
    else:
        cutoff = st.sidebar.slider("Cutoff Frequency (Hz)", 100, 2000, 800)
        order = st.sidebar.slider("Butterworth Order", 1, 6, 3)
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low')
        signal_denoised = filtfilt(b, a, signal_noisy)

    signal_denoised /= np.max(np.abs(signal_denoised))

    # -------------------------- Plots --------------------------
    st.markdown("---")
    st.subheader("Time Domain Comparison")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, signal_clean, label="Clean Signal", color="gold")
    ax.plot(t, signal_noisy, label="Noisy Signal", color="crimson", alpha=0.6)
    ax.plot(t, signal_denoised, label="Denoised", color="green", linewidth=2)
    ax.set_title("Signal Comparison")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Frequency Domain (FFT)")
    def fft_mag(sig, fs):
        N = len(sig)
        freqs = np.fft.rfftfreq(N, 1/fs)
        mag = np.abs(np.fft.rfft(sig)) / N
        mag[1:-1] *= 2
        return freqs, mag

    f_clean, m_clean = fft_mag(signal_clean, fs)
    f_noisy, m_noisy = fft_mag(signal_noisy, fs)
    f_denoised, m_denoised = fft_mag(signal_denoised, fs)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(f_clean, m_clean, label="Clean")
    ax.semilogy(f_noisy, m_noisy, label="Noisy")
    ax.semilogy(f_denoised, m_denoised, label="Denoised")
    ax.set_xlim(0, 3000 if input_mode == "Synthetic High-Freq Signal" else fs/2)
    ax.set_title("Spectrum Comparison")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    # Audio playback
    st.markdown("---")
    st.subheader("Audio Playback")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Clean")
        buf = BytesIO()
        sf.write(buf, signal_clean, fs, format="WAV")
        st.audio(buf.getvalue())
    with col2:
        st.subheader("Noisy")
        buf = BytesIO()
        sf.write(buf, signal_noisy, fs, format="WAV")
        st.audio(buf.getvalue())
    with col3:
        st.subheader("Denoised")
        buf = BytesIO()
        sf.write(buf, signal_denoised, fs, format="WAV")
        st.audio(buf.getvalue())

    st.success("**Observation**: Noise corrupts high-frequency components. Filters remove noise while preserving key frequencies (speech formants or synthetic peaks).")

else:
    st.info("Adjust parameters → click **Add Noise & Denoise** to see results!")

st.caption("DSP Lab 11 – Noise Addition & Removal from Higher Frequency Cumulative Signal (Voice Option) | 2025")