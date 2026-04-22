import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Add backend to path so we can import saliency logic
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'saliency_core'))
if backend_path not in sys.path:
    sys.path.append(backend_path)

try:
    from saliency import compute_saliency
except ImportError:
    st.error(f"Could not import saliency logic from {backend_path}. Please check your project structure.")
    st.stop()

st.set_page_config(
    page_title="Salient Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Salient Object Detection")
st.markdown("""
Extract the most prominent objects from your images using manifold ranking and superpixel segmentation.
""")

with st.sidebar:
    st.header("Settings")
    scales = st.multiselect(
        "Superpixel Scales",
        options=[300, 500, 700, 1000],
        default=[300, 500, 700],
        help="Number of superpixels to use for saliency computation at different scales."
    )
    
    bilateral_d = st.slider("Bilateral Filter Diameter", 1, 15, 9)
    bilateral_sigma_color = st.slider("Bilateral Sigma Color", 0.01, 0.5, 0.1)
    bilateral_sigma_space = st.slider("Bilateral Sigma Space", 5, 50, 15)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    if st.button("Detect Salient Objects", type="primary"):
        with st.spinner("Computing saliency maps..."):
            # Compute saliency at multiple scales
            saliency_maps = []
            for s in scales:
                saliency_s = compute_saliency(img_array, s)
                saliency_maps.append(saliency_s)
            
            if not saliency_maps:
                st.warning("Please select at least one scale.")
                st.stop()
                
            # Average saliency maps
            saliency_final = np.mean(saliency_maps, axis=0)
            
            # Post-processing
            saliency_final = saliency_final.astype(np.float32)
            saliency_final = cv2.bilateralFilter(
                saliency_final,
                d=bilateral_d,
                sigmaColor=bilateral_sigma_color,
                sigmaSpace=bilateral_sigma_space
            )
            
            # Create Pop-out effect
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            gray_float = gray_3c.astype(np.float32) / 255.0
            image_float = img_array.astype(np.float32) / 255.0
            
            saliency_3c = np.stack([saliency_final]*3, axis=-1)
            output_float = saliency_3c * image_float + (1 - saliency_3c) * gray_float
            
            output = (output_float * 255).astype(np.uint8)
            
            with col2:
                st.subheader("Saliency Result")
                st.image(output, use_container_width=True)
                
            st.divider()
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("Raw Saliency Map")
                # Normalize for display
                sal_disp = (saliency_final * 255).astype(np.uint8)
                st.image(sal_disp, use_container_width=True, channels="L")
            
            with res_col2:
                st.subheader("Heatmap")
                # Apply colormap to saliency
                heatmap = cv2.applyColorMap((saliency_final * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                st.image(heatmap, use_container_width=True)

else:
    st.info("Please upload an image to begin.")
    
    # Show examples if available
    example_dir = os.path.join(backend_path, "Test Images")
    if os.path.exists(example_dir):
        st.subheader("Example Images")
        examples = [f for f in os.listdir(example_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if examples:
            cols = st.columns(min(len(examples), 4))
            for i, ex in enumerate(examples[:4]):
                ex_path = os.path.join(example_dir, ex)
                ex_img = Image.open(ex_path)
                cols[i].image(ex_img, caption=ex, use_container_width=True)
