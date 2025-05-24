import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

st.set_page_config(page_title="TrOCR Text Recognition Comparison", layout="wide")

@st.cache_resource
def load_models():
    processor_pretrained = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model_pretrained = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    
    processor_finetuned = TrOCRProcessor.from_pretrained("saved_model_version2")
    model_finetuned = VisionEncoderDecoderModel.from_pretrained("saved_model_version2")
    
    return processor_pretrained, model_pretrained, processor_finetuned, model_finetuned

processor_pretrained, model_pretrained, processor_finetuned, model_finetuned = load_models()

st.title("Handwritten Text Recognition: Pre-trained vs Fine-Tuned TrOCR")
st.write("Upload an image to extract text and compare results from both the pre-trained and fine-tuned TrOCR models.")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Extract Text"):
        with st.spinner("Processing with both models..."):
            pixel_values_pretrained = processor_pretrained(image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids_pretrained = model_pretrained.generate(pixel_values_pretrained)
            text_pretrained = processor_pretrained.batch_decode(generated_ids_pretrained, skip_special_tokens=True)[0]
            
            pixel_values_finetuned = processor_finetuned(image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids_finetuned = model_finetuned.generate(pixel_values_finetuned)
            text_finetuned = processor_finetuned.batch_decode(generated_ids_finetuned, skip_special_tokens=True)[0]

            st.subheader("Extracted Text:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Pre-trained Model Output:**")
                st.write(text_pretrained)
            with col2:
                st.write("**Fine-tuned Model Output:**")
                st.write(text_finetuned)