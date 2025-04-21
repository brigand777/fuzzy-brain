import streamlit as st
from utils.glossary import add_glossary_term, GLOSSARY_TERMS

st.set_page_config(page_title="📚 Glossary", layout="wide")
st.title("📚 Portfolio Tuner Glossary")
st.markdown("Get familiar with key terms used throughout the app. Click on a term to learn more.")
# --- Custom CSS to Increase Font Size for Normal Text ---
st.markdown("""
<style>
/* Bump up normal text font size */
.stMarkdown p, .stMarkdown div, p {
    font-size: 18px !important;
    line-height: 1.5;
}

/* Restore proper sizes for headers */
h1 {
    font-size: 36px !important;
    font-weight: 700;
}
h2 {
    font-size: 28px !important;
    font-weight: 600;
}
h3 {
    font-size: 22px !important;
    font-weight: 600;
}

/* Preserve table/chart styling */
.stDataFrame, .stTable, .stPlotlyChart, .stAltairChart {
    font-size: inherit !important;
}

/* Optional: Larger narrative box text */
.narrative-box {
    font-size: 20px !important;
}
</style>
""", unsafe_allow_html=True)
for item in GLOSSARY_TERMS:
    add_glossary_term(item["term"], item["short"], item["long"])
