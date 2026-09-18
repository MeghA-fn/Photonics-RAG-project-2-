import streamlit as st
import requests

# -----------------------------------------------------
# 13.1 Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Photonics RAG",
    page_icon="🔬",
    layout="wide"
)

# -----------------------------------------------------
# Title
# -----------------------------------------------------
st.title("Photonics RAG Assistant")

st.write("Ask questions related to photonics")

# -----------------------------------------------------
# 13.3 Sidebar
# -----------------------------------------------------
st.sidebar.title("About")

st.sidebar.info("""
This AI assistant answers the questions
related to Photonics world
""")

# -----------------------------------------------------
# User Input
# -----------------------------------------------------
question = st.text_input("Ask a Photonics question:")

# -----------------------------------------------------
# Ask Button
# -----------------------------------------------------
if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents and generating answer..."):

            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={
                    "question": question
                }
            )

            result = response.json()

        # ------------------------------------------------
        # No Answer Case
        # ------------------------------------------------
        if result["status"] == "no_answer_found":

            st.error(result["message"])

            st.subheader("Documents Checked")

            for source in result["sources"]:

                pages = ", ".join(str(p) for p in source["pages"])

                st.write(f"📄 {source['document']} (Pages: {pages})")

        # ------------------------------------------------
        # Success Case
        # ------------------------------------------------
        else:

            # 13.8 Better Answer Box
            st.subheader("💡 Answer")
            st.info(result["answer"])

            # 13.7 Confidence Metric
            st.subheader("Confidence")
            confidence = result["confidence"]

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.4f}"
            )

            st.subheader("📄 Sources")

            for source in result["sources"]:

                pages = ", ".join(str(p) for p in source["pages"])

                st.write(f"📄 {source['document']} (Pages: {pages})")

# -----------------------------------------------------
# 13.11 Footer
# -----------------------------------------------------
st.markdown("---")

st.caption(
    "| Photonics RAG System | "
)