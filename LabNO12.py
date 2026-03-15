# LABNo12_Record_Voice_Moving_Average_Final_With_Spectrograms.py – Final Version (2025)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO

st.set_page_config(page_title="DSP Lab 12 – Live Voice Denoising", layout="wide")
st.title("DSP Lab 12 – Record Voice → Add Noise → Moving Average Filter (3–12 pt)")
st.markdown("**Record directly → Add noise → Progressive denoising + Final comparisons including spectrograms**")

# -------------------------- Record Voice Directly --------------------------
st.subheader("Step 1: Record Your Voice (5–15 seconds)")
audio_file = st.audio_input("Click mic → speak clearly → click stop", key="recorder")

if audio_file is not None:
    audio_bytes = audio_file.read()
    clean_np, fs = sf.read(BytesIO(audio_bytes))
    
    if clean_np.ndim > 1:
        clean_np = np.mean(clean_np, axis=1)
    
    clean_np = clean_np / (np.max(np.abs(clean_np)) + 1e-9)
    duration = len(clean_np) / fs
    t = np.linspace(0, duration, len(clean_np))

    st.audio(audio_bytes)
    st.success(f"Recorded: {duration:.2f} seconds @ {fs:,} Hz")

    # -------------------------- Add Noise --------------------------
    noise_level = st.slider("Gaussian Noise Level", 0.01, 0.60, 0.20, 0.01)

    if st.button("Add Noise & Apply Filters", type="primary", use_container_width=True):
        with st.spinner("Adding noise..."):
            np.random.seed()
            noise = np.random.normal(0, noise_level, len(clean_np))
            noisy_np = clean_np + noise
            noisy_np = noisy_np / (np.max(np.abs(noisy_np)) + 1e-9)

            buf = BytesIO()
            sf.write(buf, noisy_np, fs, format="WAV")
            noisy_bytes = buf.getvalue()

            st.session_state.update({
                "clean_np": clean_np,
                "noisy_np": noisy_np,
                "noisy_bytes": noisy_bytes,
                "fs": fs,
                "t": t,
                "clean_bytes": audio_bytes
            })
        st.success("Noise added! Applying filters...")
        st.rerun()

# -------------------------- Results --------------------------
if "noisy_np" in st.session_state:
    clean_np = st.session_state.clean_np
    noisy_np = st.session_state.noisy_np
    fs = st.session_state.fs
    t = st.session_state.t
    clean_bytes = st.session_state.clean_bytes
    noisy_bytes = st.session_state.noisy_bytes

    # Noisy version
    st.markdown("---")
    st.subheader("🔴 Noisy Voice")
    c1, c2 = st.columns(2)
    with c1: 
        st.subheader("Audio")
        st.audio(noisy_bytes)
    with c2:
        fig, ax = plt.subplots()
        ax.plot(t, noisy_np, color="crimson")
        ax.set_title("Noisy Waveform")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("🟢 Progressive Moving Average Filtering")

    best_filtered = noisy_np
    for M in range(3, 13):
        kernel = np.ones(M) / M
        filtered = np.convolve(best_filtered, kernel, mode='same')
        filtered = filtered / (np.max(np.abs(filtered)) + 1e-9)

        buf = BytesIO()
        sf.write(buf, filtered, fs, format="WAV")
        filtered_bytes = buf.getvalue()

        st.markdown(f"### {M}-Point Moving Average")
        cols = st.columns([1.2, 2, 2])
        with cols[0]:
            st.subheader("Audio")
            st.audio(filtered_bytes)
        with cols[1]:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(t, noisy_np, color="lightcoral", alpha=0.5, label="Noisy")
            ax.plot(t, filtered, color="green", linewidth=2, label=f"{M}-pt")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        with cols[2]:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.specgram(filtered, NFFT=1024, Fs=fs, noverlap=512, cmap="viridis")
            ax.set_ylim(0, 8000)
            ax.set_title(f"Spectrogram ({M}-pt)")
            st.pyplot(fig)
            plt.close(fig)

        best_filtered = filtered

    # -------------------------- FINAL COMPARISON --------------------------
    st.markdown("---")
    st.subheader("🔵 FINAL COMPARISON: Original Clean vs Best Filtered (12-point)")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Clean Audio")
        st.audio(clean_bytes)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(t, clean_np, color="steelblue", linewidth=2)
        ax.set_title("Original Clean Waveform")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("After 12-point Filter Audio")
        buf = BytesIO()
        sf.write(buf, best_filtered, fs, format="WAV")
        st.audio(buf.getvalue())
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(t, best_filtered, color="darkgreen", linewidth=2)
        ax.set_title("Best Filtered Waveform (12-pt)")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Overlaid waveform
    st.markdown("#### Overlaid Waveform – Original vs 12-point Filtered")
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(t, clean_np, color="steelblue", linewidth=2, label="Original Clean", alpha=0.8)
    ax.plot(t, best_filtered, color="darkgreen", linewidth=2, label="12-point Filtered", alpha=0.8)
    ax.set_title("Original vs 12-point Moving Average Filtered (Overlaid)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # NEW: Side-by-side Spectrograms of Original and Final Filtered
    st.markdown("#### Final Spectrogram Comparison – Original vs 12-point Filtered")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spectrogram: Original Clean")
        fig, ax = plt.subplots(figsize=(12,6))
        Pxx, freqs, bins, im = ax.specgram(clean_np, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
        ax.set_ylim(0, 8000)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax, label="Intensity (dB)")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Spectrogram: After 12-point Filter")
        fig, ax = plt.subplots(figsize=(12,6))
        Pxx, freqs, bins, im = ax.specgram(best_filtered, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
        ax.set_ylim(0, 8000)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax, label="Intensity (dB)")
        st.pyplot(fig)
        plt.close(fig)

    st.success("**Conclusion**: Moving average filter removes high-frequency noise (visible as reduced 'fog' in spectrogram) while preserving speech formants!")

    if st.button("Record New Voice & Start Over", type="secondary", use_container_width=True):
        keys = ["clean_np","noisy_np","noisy_bytes","fs","t","clean_bytes"]
        for k in keys:
            st.session_state.pop(k, None)
        st.rerun()

else:
    st.info("Click the mic → record → adjust noise → click **Add Noise & Apply Filters**")
    st.balloons()

st.caption("DSP Lab 12 – Live Recording + Moving Average Denoising + Final Waveform & Spectrogram Comparison | 2025")