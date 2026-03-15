# LABNo14_Image_Enhancement_Fixed.py – Image Enhancement (Fixed Deprecation Warnings)
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

st.set_page_config(page_title="DSP Lab 14 – Image Enhancement", layout="wide")
st.title("DSP Lab 14 – Image Enhancement using Gamma (Beta), Intensity & Contrast")
st.markdown("**Upload image → Adjust parameters → See real-time changes**")

# -------------------------- Upload Image --------------------------
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    img_pil = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    st.success(f"Image loaded: {img_cv.shape[1]}×{img_cv.shape[0]} pixels")
    
    # Display original
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_pil, width=600)  # Fixed: use width instead of use_column_width

    # -------------------------- Enhancement Parameters --------------------------
    st.markdown("---")
    st.subheader("Enhancement Controls")

    beta = st.slider("Beta (Gamma Correction)", 0.1, 3.0, 1.0, 0.1, 
                     help="Beta < 1 → brighten dark areas\nBeta > 1 → darken")
    intensity = st.slider("Intensity (Brightness)", -100, 100, 0, 10, 
                          help="Positive → brighter\nNegative → darker")
    contrast = st.slider("Contrast Factor", 0.5, 3.0, 1.0, 0.1, 
                         help=">1 → increase contrast\n<1 → decrease contrast")

    # -------------------------- Apply Enhancements --------------------------
    # Step 1: Gamma correction (Beta)
    img_float = img_cv / 255.0
    img_gamma = np.power(img_float, beta)
    
    # Step 2: Intensity (Brightness)
    img_intensity = img_gamma + (intensity / 255.0)
    img_intensity = np.clip(img_intensity, 0, 1)
    
    # Step 3: Contrast
    img_contrast = (img_intensity - 0.5) * contrast + 0.5
    img_contrast = np.clip(img_contrast, 0, 1)
    
    # Convert back to uint8
    enhanced_cv = (img_contrast * 255).astype(np.uint8)
    enhanced_rgb = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB)
    enhanced_pil = Image.fromarray(enhanced_rgb)

    # Display enhanced
    with col2:
        st.subheader("Enhanced Image")
        st.image(enhanced_pil, width=600)  # Fixed: use width instead of use_column_width

    # -------------------------- Side-by-Side Comparison --------------------------
    st.markdown("---")
    st.subheader("Comparison: Original vs Enhanced")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_pil, caption="Original", width=600)  # Fixed
    with col2:
        st.image(enhanced_pil, caption="Enhanced", width=600)  # Fixed

    # Download enhanced image
    buf = BytesIO()
    enhanced_pil.save(buf, format="PNG")
    st.download_button("Download Enhanced Image", buf.getvalue(), "enhanced_image.png", "image/png")

else:
    st.info("👆 Upload an image to start enhancement!")
    st.markdown("""
    **What each parameter does:**
    - **Beta (Gamma)**: Adjusts brightness non-linearly (great for underexposed images)
    - **Intensity**: Simple brightness shift
    - **Contrast**: Expands/reduces the dynamic range of pixel values
    
    Try different images (dark, bright, low-contrast) to see effects!
    """)

st.caption("DSP Lab 14 – Image Enhancement: Gamma (Beta), Intensity & Contrast | Fixed Deprecation Warnings | 2025")