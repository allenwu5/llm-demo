"""
refer to https://github.com/avantrio/chroma-viewer/blob/main/chroma-viewer/viewer.py
"""

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


import chromadb
import pandas as pd
import streamlit as st




def view_collections(db):
    st.header("DB: %s" % db)

    host, port = db.split(":")
    client = chromadb.HttpClient(host=host, port=port)

    for collection in client.list_collections():
        data = collection.get()

        ids = data["ids"]
        embeddings = data["embeddings"]
        metadata = data["metadatas"]
        documents = data["documents"]

        df = pd.DataFrame.from_dict(data)
        st.subheader("Collection: **%s**" % collection.name)
        st.dataframe(df)

view_collections("chromadb:8000")
