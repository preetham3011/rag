"""Document upload interface"""

import streamlit as st


def render_upload_widget():
    """Render document upload widget"""
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or HTML file",
        type=["pdf", "html"]
    )
    return uploaded_file
