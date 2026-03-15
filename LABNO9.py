# LABNo9_Noise_Removal_RANDOM.py – Realistic Random Noise + Removal (2025)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO

st.set_page_config(page_title="DSP Lab 9 – Realistic Noise Removal", layout="wide")
st.title("DSP Lab 9 – Random Noise Addition & Removal")
st.markdown("**Real-world scenario: Random noise every time + noise estimated from silence**")

# -------------------------- Sidebar --------------------------
st.sidebar.header("Parameters")
noise_level = st.sidebar.slider("Random Noise Strength", 0.02, 0.35, 0.10, 0.01)
alpha = st.sidebar.slider("Over-subtraction factor (α)", 1.0, 6.0, 3.0, 0.2)
beta  = st.sidebar.slider("Spectral floor (β)", 0.001, 0.1, 0.02, 0.001)

st.sidebar.info("Noise changes every time you click → Just like real life!")

# -------------------------- Record Audio --------------------------
st.subheader("Step 1: Record Your Voice (5–10 seconds)")
st.info("Start speaking after 1 second of silence so we can estimate noise")
audio_file = st.audio_input("Click mic → wait 1s → speak → stop", key="recorder")

if audio_file is not None:
    # Load
    audio_bytes = audio_file.read()
    clean_np, fs = sf.read(BytesIO(audio_bytes))
    if clean_np.ndim > 1:
        clean_np = np.mean(clean_np, axis=1)
    clean_np = clean_np / (np.max(np.abs(clean_np)) + 1e-9)

    duration = len(clean_np) / fs
    st.success(f"Recorded {duration:.2f} seconds @ {fs:,} Hz")
    st.audio(audio_bytes, format="audio/wav")

    if st.button("Add RANDOM Noise → Remove It!", type="primary", use_container_width=True):
        with st.spinner("Generating new random noise + denoising..."):
            # NEW RANDOM NOISE every single time!
            np.random.seed()  # ensures truly different noise each run
            noise = np.random.normal(0, noise_level, len(clean_np))
            noisy_np = clean_np + noise
            noisy_np = noisy_np / (np.max(np.abs(noisy_np)) + 1e-9)

            # === REALISTIC NOISE ESTIMATION ===
            # Take first ~150 ms as noise-only (common in real speech recordings)
            noise_samples = int(0.15 * fs)
            noise_estimate_segment = noisy_np[:noise_samples]

            # STFT parameters
            N = 1024
            hop = N // 4
            window = np.hanning(N)

            # Compute STFT of noisy signal
            frames = [noisy_np[i:i+N] for i in range(0, len(noisy_np)-N+1, hop)]
            stft_frames = [np.fft.rfft(window * frame) for frame in frames]
            mag_noisy = np.abs(np.array(stft_frames))
            phase = np.angle(np.array(stft_frames))

            # Estimate noise spectrum from the silent beginning
            noise_mag_est = np.mean(mag_noisy[:8], axis=0)  # first ~8 frames ≈ 150–200ms

            # Spectral Subtraction with parameters
            mag_clean = mag_noisy - alpha * noise_mag_est
            mag_clean = np.maximum(mag_clean, beta * mag_noisy)  # spectral floor

            # Reconstruct
            clean_stft = mag_clean * np.exp(1j * phase)
            denoised_np = np.zeros(len(noisy_np))

            for i, frame in enumerate(clean_stft):
                time_frame = np.fft.irfft(frame)
                start = i * hop
                denoised_np[start:start+N] += time_frame * window

            # Normalize
            denoised_np /= (np.max(np.abs(denoised_np)) + 1e-9)

            # Save denoised audio
            buffer = BytesIO()
            sf.write(buffer, denoised_np, fs, format="WAV")
            denoised_bytes = buffer.getvalue()

            # Store in session
            st.session_state.update({
                "clean_np": clean_np,
                "noisy_np": noisy_np,
                "denoised_np": denoised_np,
                "fs": fs,
                "clean_bytes": audio_bytes,
                "denoised_bytes": denoised_bytes
            })
        st.success("Random noise added & successfully removed!")
        st.balloons()
        st.rerun()

# -------------------------- RESULTS (3-way comparison) --------------------------
if "denoised_np" in st.session_state:
    c = st.session_state.clean_np
    n = st.session_state.noisy_np
    d = st.session_state.denoised_np
    fs = st.session_state.fs

    time = np.linspace(0, len(c)/fs, len(c))
    freqs = np.fft.rfftfreq(len(c), 1/fs)

    # Audio Players
    col1, col2, col3, col4 = st.columns([1,1,1,2])
    with col1:
        st.subheader("Original Clean")
        st.audio(st.session_state.clean_bytes, format="audio/wav")
    with col2:
        st.subheader("Noisy (Random!)")
        buf = BytesIO()
        sf.write(buf, n, fs, format="WAV")
        st.audio(buf.getvalue(), format="audio/wav")
    with col3:
        st.subheader("Denoised")
        st.audio(st.session_state.denoised_bytes, format="audio/wav")
    with col4:
        st.download_button("Download Denoised.wav", st.session_state.denoised_bytes,
                           "denoised_clean_voice.wav", "audio/wav", use_container_width=True)

    st.markdown("---")

    # Waveforms
    st.subheader("Time Domain")
    cols = st.columns(3)
    for col, data, label, color in zip(cols, [c, n, d], ["Clean", "Noisy", "Denoised"], ["steelblue", "crimson", "green"]):
        with col:
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(time, data, color=color, linewidth=1)
            ax.set_title(label, fontweight="bold", fontsize=13)
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)

    # FFT
    st.subheader("Frequency Domain (Magnitude Spectrum)")
    cols = st.columns(3)
    for col, data, label, color in zip(cols, [c, n, d], ["Clean", "Noisy", "Denoised"], ["steelblue", "crimson", "green"]):
        with col:
            fig, ax = plt.subplots(figsize=(8,3))
            ax.semilogy(freqs, np.abs(np.fft.rfft(data)), color=color)
            ax.set_xlim(0, 8000)
            ax.set_title(label, fontweight="bold")
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)

    # Spectrograms
    st.subheader("Spectrograms")
    cols = st.columns(3)
    for col, data, label in zip(cols, [c, n, d], ["Clean", "Noisy", "Denoised"]):
        with col:
            fig, ax = plt.subplots(figsize=(8,4))
            Pxx, freq, t, im = ax.specgram(data, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
            ax.set_ylim(0, 8000)
            ax.set_title(label, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, use_container_width=True)

    if st.button("Try Again", type="secondary", use_container_width=True):
        for k in st.session_state.keys():
            if k.startswith(("clean", "noisy", "denoised", "fs")):
                st.session_state.pop(k)
        st.rerun()

else:
    st.info("Record → Wait 1 second silence → Speak → Click the button → Watch magic happen!")
    st.markdown("**Noise is 100% random every time — exactly like real-world conditions!**")

st.caption("DSP Lab 9 – Realistic Random Noise Removal using Spectral Subtraction | 2025")