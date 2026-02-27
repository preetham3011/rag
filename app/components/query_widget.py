"""Query input and intent display"""

import streamlit as st


def render_query_widget():
    """Render query input widget"""
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question:")
    return query


def display_intent(intent: str):
    """Display detected query intent"""
    st.info(f"Detected Intent: {intent}")
