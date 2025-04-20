import streamlit as st
from utils.glossary import add_glossary_term, GLOSSARY_TERMS

st.set_page_config(page_title="📚 Glossary", layout="wide")
st.title("📚 Portfolio Tuner Glossary")
st.markdown("Get familiar with key terms used throughout the app. Click on a term to learn more.")

for item in GLOSSARY_TERMS:
    add_glossary_term(item["term"], item["short"], item["long"])
