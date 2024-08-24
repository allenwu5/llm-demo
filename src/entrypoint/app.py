import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader

sys.path.append("/workspaces")


from os import environ

from src.service.rag import Rag
from src.service.vectordb import VectorDB

COLLECTION = environ["COLLECTION"]


def main():
    db = VectorDB(COLLECTION)
    if st.sidebar.button("Clean"):
        db.delete_collection()
        db = VectorDB(COLLECTION)
        for f in Path("/upload").iterdir():
            f.unlink()

    rag = Rag(db)

    st.header("打造第二大腦")
    st.subheader("AI 數位大腦，無所不及")

    uploaded = st.file_uploader("上傳 PDF", type=["pdf"])
    if uploaded is not None:
        f = Path("/upload", uploaded.name)
        if not f.exists():
            path = str(f)
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
                with st.spinner("建立索引"):
                    loader = PyMuPDFLoader(path)
                    docs = loader.load()
                    db.index(docs)

    files = []
    for p in Path("/upload").iterdir():
        files.append(p.name)
    df = pd.DataFrame(files, columns=["已索引檔案"])
    st.sidebar.dataframe(df, hide_index=True)

    query = st.radio(
        "範例問題",
        ("無", "上班的服裝費可以抵稅嗎？", "不繳遺產稅會？"),
        horizontal=True,
    )

    if query == "無":
        query = ""
    query = st.text_area(
        "輸入問題:",
        query,
    )
    submitted = st.button("送出")

    if submitted:
        docs = db.retrieval(query)
        if docs:
            doc_top1, score_top1 = docs[0]
        else:
            doc_top1, score_top1 = None, 0
        debug_info = doc_top1

        with st.spinner("生成中"):
            ans = "VectorDB 的資料不足以提供答案，請提供更詳細的問題。"
            if score_top1 > 0.5:
                res = rag.predict(query)
                debug_info = res
                ans = res["answer"]

            st.subheader("回答")
            st.write(ans)

            st.subheader("參考資料")
            score_top1_percentage = round(score_top1 * 100, 1)
            st.metric("參考資料相似度", f"{score_top1_percentage}%")
            
            with st.expander("看參考資料"):
                st.code(doc_top1)

            with st.expander("Debug Info."):
                st.code(debug_info)

    # st.subheader("問題")


if __name__ == "__main__":
    main()
