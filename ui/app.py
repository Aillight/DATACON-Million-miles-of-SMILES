import json
import os
import urllib.error
import urllib.request

import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


st.set_page_config(page_title="DataCon AI Agent MVP", layout="wide")
st.title("DataCon AI Agent MVP")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None and st.button("Parse PDF"):
    boundary = "----datacon-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{uploaded_file.name}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8")
    body += uploaded_file.getvalue()
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    request = urllib.request.Request(
        f"{BACKEND_URL}/parse/pdf",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        st.subheader("Markdown")
        st.text_area("Parsed content", result.get("markdown", ""), height=500)
        if result.get("warnings"):
            st.warning("\n".join(result["warnings"]))
    except urllib.error.URLError as exc:
        st.error(f"Backend request failed: {exc}")
