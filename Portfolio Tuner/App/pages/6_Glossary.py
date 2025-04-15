import streamlit as st
from utils.glossary import add_glossary_term, GLOSSARY_TERMS

st.set_page_config(page_title="📚 Glossary", layout="wide")
st.title("📚 Portfolio Tuner Glossary")
st.markdown("Get familiar with key terms used throughout the app. Click on a term to learn more.")

for item in GLOSSARY_TERMS:
    add_glossary_term(item["term"], item["short"], item["long"])

st.markdown("### 🔤 Font Preview")
fonts = {
    "Lora": "https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;1,400&display=swap",
    "Merriweather": "https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,400;1,400&display=swap",
    "EB Garamond": "https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;1,400&display=swap",
    "Cormorant Garamond": "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;1,400&display=swap",
}

for name, link in fonts.items():
    st.markdown(f'<link href="{link}" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:\'{name}\', serif; font-style:italic;">This is {name} — a sample of tooltip text style.</p>', unsafe_allow_html=True)
