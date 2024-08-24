
import streamlit as st
import sys
import pandas as pd
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader


sys.path.append("/workspaces")


from src.service.rag import Rag
from src.service.vectordb import VectorDB
from os import environ

COLLECTION=environ["COLLECTION"]

def main():
    db = VectorDB(COLLECTION)
    if st.sidebar.button("Clean"):
        db.delete_collection()
        db = VectorDB(COLLECTION)
        for f in Path("/upload").iterdir():
            f.unlink()
        
    rag = Rag(db)
    
    st.header("大型語言模型實作班 第四期 第 3 組 DEMO ")
    
    uploaded = st.file_uploader("上傳 PDF",type=['pdf'])
    if uploaded is not None:
        f = Path("/upload",uploaded.name)
        if not f.exists():
            path = str(f)
            with open(path,"wb") as f: 
                f.write(uploaded.getbuffer())   
                with st.spinner('建立索引'):
                    loader = PyMuPDFLoader(path)
                    docs = loader.load()
                    db.index(docs)

    files = []
    for p in Path("/upload").iterdir():
        files.append(p.name)
    df = pd.DataFrame(
        files, columns=["已索引檔案"]
    )
    st.sidebar.dataframe(df, hide_index=True)
   
    st.subheader("問題")
    query = st.radio(
    "範例問題",
    ("無", "上班的服裝費可以抵稅嗎？", "不繳遺產稅會？"),horizontal=True
)
    if query == "無":
        query = st.text_input("用 Enter 送出題目", "", placeholder="請填入題目")
    
    if query:
        docs = db.retrieval(query)
        if docs:
            doc_top1, score_top1 = docs[0]
        else:
            doc_top1, score_top1 = None, 0
        debug_info = doc_top1
        
        with st.spinner('生成中'):
            ans = "VectorDB 的資料不足以提供答案，請提供更詳細的問題。"
            if score_top1 > 0.7:
                res = rag.predict(query)
                debug_info = res
                ans = res["answer"]

            st.subheader("回答")
            st.write(ans)
            
            with st.expander("看參考資料"):
                st.subheader("參考資料 score")
                st.write(score_top1)
                
                st.subheader("參考資料")
                st.write(doc_top1)
            
            with st.expander("See debug info"):
                st.write(debug_info)
    


if __name__ == "__main__":
    main()
