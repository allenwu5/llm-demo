__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from tqdm import tqdm

from src.config import *

from .embedding import embedding_func


# https://github.com/chroma-core/chroma/issues/1049
def split_docs_to_batches(docs, batch_size):
    for i in range(0, len(docs), batch_size):
        yield docs[i : i + batch_size]


class VectorDB:
    def __init__(self, collection_name) -> None:
        self.collection_name = collection_name
        # Initialize the client with the remote URL
        self.client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT,
            ssl=False,
            headers=None,
            settings=Settings(),
            # tenant=DEFAULT_TENANT,
            # database=DEFAULT_DATABASE,
        )
        self.db = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedding_func,
        )

    def delete_collection(self):
        if self.collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(self.collection_name)

    def index(self, docs):
        batches = 32
        doc_batches = list(split_docs_to_batches(docs, batches))

        for doc_batch in tqdm(doc_batches):
            self.add_docs(doc_batch)

    def add_docs(self, docs):
        Chroma.from_documents(
            docs,
            embedding_func,
            client=self.client,
            collection_name=self.collection_name,
        )

    def retrieval(self, query):
        return self.db.similarity_search_with_relevance_scores(query)
