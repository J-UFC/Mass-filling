import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Masked Word Filler (MLM)", page_icon="🧠")
st.title("🧠 Masked Word Filler (MLM)")
st.write("Enter a sentence with a `[MASK]` token to let the model fill in the blank.")

@st.cache_resource
def load_model():
    return pipeline("fill-mask", model="bert-base-uncased")

fill_mask = load_model()

text = st.text_input("Enter your sentence with [MASK]:", value="Elon [MASK] is the CEO of Tesla.")

if text and "[MASK]" in text:
    with st.spinner("Predicting..."):
        results = fill_mask(text)
        st.subheader("Predictions:")
        for res in results:
            st.write(f"🔹 {res['sequence']} (score: {res['score']:.4f})")
else:
    st.info("ℹ️ Please enter a sentence containing a [MASK] token.")