"""
app_with_uploader.py
Alternative entry point — allows uploading Insurance.csv via the UI.
Use this if you don't want to commit the CSV to GitHub.
"""
import streamlit as st
import os

st.set_page_config(page_title="Insurance Bias Analysis", page_icon="🔍", layout="wide")

uploaded = st.file_uploader("Upload Insurance.csv to begin", type=["csv"])

if uploaded:
    import tempfile, shutil
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(uploaded.read())
    tmp.flush()
    # Copy to app root so app.py can find it
    shutil.copy(tmp.name, "Insurance.csv")
    st.success("File uploaded! Loading dashboard…")
    # Re-run the main app
    exec(open("app.py").read())
else:
    st.info("Please upload the Insurance.csv file to launch the full analysis dashboard.")
