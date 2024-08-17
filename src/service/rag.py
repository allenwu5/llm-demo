from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from .llm import llm

PROMPT_TEMPLATE = """請用以下資訊來回答問題：

{context}

Question: {question}
Answer(Let's think step by step):"""


class Rag:
    def __init__(self, vector_db) -> None:
        self.db = vector_db.db
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": prompt}

        # self.qa = RetrievalQA.from_chain_type(
        #     llm=llm_taide_llama2_7b,
        #     chain_type="stuff",
        #     retriever=self.retriever,
        #     chain_type_kwargs=chain_type_kwargs,
        #     return_source_documents=True,
        # )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
        )
        self.qa = ConversationalRetrievalChain.from_llm(
            llm,
            self.retriever,
            memory=memory,
            combine_docs_chain_kwargs=chain_type_kwargs,
            return_source_documents=True,
        )

    def predict(self, query):
        return self.qa.invoke(query)
