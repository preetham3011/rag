"""Display compressed evidence, answer, and citations"""

import streamlit as st


def display_results(result: dict):
    """
    Display query results
    
    Args:
        result: dict with 'intent', 'chunks', 'compressed_evidence', 
                'answer', 'citations', 'refusal'
    """
    if result.get("refusal"):
        st.warning(result["refusal"])
        return
    
    with st.expander("Retrieved Chunks"):
        st.write(result.get("chunks", []))
    
    with st.expander("Compressed Evidence"):
        st.write(result.get("compressed_evidence", ""))
    
    st.subheader("Answer")
    st.write(result.get("answer", ""))
    
    if result.get("citations"):
        st.subheader("Citations")
        st.write(result["citations"])
