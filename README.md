# llm-demo

# 設定
## OpenAI API 設定
`secrets/openai_api_key.json`
```
{
    "OPENAI_API_KEY": "...",
    "OPENAI_ORGANIZATION": "..."
}
```

# LLM APP
## Web APP (Streamlit)
```python
import streamlit as st
```

# 檢索
## Embedding (OpenAIEmbeddings)
```python
from langchain_openai import OpenAIEmbeddings
embedding_func = OpenAIEmbeddings()
```

## Vector DB (ChromaDB)
```python
import chromadb
from langchain_chroma import Chroma

class VectorDB:
    def __init__(self, collection_name) -> None:
        self.collection_name = collection_name
        self.client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT,
            ssl=False,
            headers=None,
            settings=Settings(),
        )
        self.db = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedding_func,
        )
    ...
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
```

## 對文件建立索引
上傳 PDF 檔案：Document loader (PyMuPDFLoader)
```python
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader

db = VectorDB("demo")
...
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

```

# 檢索增強生成（Retrieval-Augmented Generation, RAG)
## 生成 by LLM
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
```

## Prompt
```python
PROMPT_TEMPLATE = """請用以下資訊來回答問題：

{context}

Question: {question}
Answer(Let's think step by step):"""
```

## RAG
```python
class Rag:
    def __init__(self, vector_db) -> None:
        self.db = vector_db.db
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": prompt}

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
```

# Demo
Query:
```python
query = "上班的服裝費可以抵稅嗎？"
res = rag.predict(query)
```

Answer:
```
根據提供的資訊，上班的服裝費是可以抵稅的，但有一些條件和限制：

職業專用服裝費：如果你所穿著的服裝是職業所必需的特殊服裝或表演專用服裝，那麼這些服裝的購置、租用、清潔及維護費用可以作為必要費用來抵稅。

減除金額限制：每人全年可以減除的金額以其從事該職業薪資收入總額的百分之三為限。這意味著你可以抵稅的服裝費用不可以超過你薪資收入的3%。

證明文件：需要有相關的證明文件來核實這些費用，以便在報稅時能夠正確地申報。

總結來說，如果你的服裝符合職業需求，並且你能提供相應的證明文件，那麼上班的服裝費是可以抵稅的，但需遵循上述的限制和規定。
```

Response Detail
```json
{
  "question": "上班的服裝費可以抵稅嗎？",
  "chat_history": [
    "HumanMessage(content='上班的服裝費可以抵稅嗎？')",
    "AIMessage(content='根據提供的資訊，上班的服裝費是可以抵稅的，但有一些條件和限制：\\n\\n1. **職業專用服裝費**：如果你所穿著的服裝是職業所必需的特殊服裝或表演專用服裝，那麼這些服裝的購置、租用、清潔及維護費用可以作為必要費用來抵稅。\\n\\n2. **減除金額限制**：每人全年可以減除的金額以其從事該職業薪資收入總額的百分之三為限。這意味著你可以抵稅的服裝費用不可以超過你薪資收入的3%。\\n\\n3. **證明文件**：需要有相關的證明文件來核實這些費用，以便在報稅時能夠正確地申報。\\n\\n總結來說，如果你的服裝符合職業需求，並且你能提供相應的證明文件，那麼上班的服裝費是可以抵稅的，但需遵循上述的限制和規定。')"
  ],
  "answer": "根據提供的資訊，上班的服裝費是可以抵稅的，但有一些條件和限制：\n\n1. **職業專用服裝費**：如果你所穿著的服裝是職業所必需的特殊服裝或表演專用服裝，那麼這些服裝的購置、租用、清潔及維護費用可以作為必要費用來抵稅。\n\n2. **減除金額限制**：每人全年可以減除的金額以其從事該職業薪資收入總額的百分之三為限。這意味著你可以抵稅的服裝費用不可以超過你薪資收入的3%。\n\n3. **證明文件**：需要有相關的證明文件來核實這些費用，以便在報稅時能夠正確地申報。\n\n總結來說，如果你的服裝符合職業需求，並且你能提供相應的證明文件，那麼上班的服裝費是可以抵稅的，但需遵循上述的限制和規定。",
  "source_documents": [
    "Document(metadata={'author': '全國法規資料庫', 'creationDate': 'D:20200927120100Z', 'creator': 'Microsoft Office Word', 'file_path': '/upload/所得稅法.pdf', 'format': 'PDF 1.7', 'keywords': '', 'modDate': 'D:20240531012700Z', 'page': 8, 'producer': 'Aspose.Words for .NET 24.5.0', 'source': '/upload/所得稅法.pdf', 'subject': '', 'title': '所得稅法', 'total_pages': 49, 'trapped': ''}, page_content='相關證明文件核實自薪資收入中減除該必要費用，以其餘額為所得額：\\n（一）職業專用服裝費：職業所必需穿著之特殊服裝或表演專用服裝，其購置、租用、清潔\\n及維護費用。每人全年減除金額以其從事該職業薪資收入總額之百分之三為限。\\n（二）進修訓練費：參加符合規定之機構開設職務上、工作上或依法令要求所需特定技能或\\n專業知識相關課程之訓練費用。每人全年減除金額以其薪資收入總額之百分之三為\\n限。\\n（三）職業上工具支出：購置專供職務上或工作上使用書籍、期刊及工具之支出。但其效能\\n非二年內所能耗竭且支出超過一定金額者，應逐年攤提折舊或攤銷費用。每人全年減\\n除金額以其從事該職業薪資收入總額之百分之三為限。\\n二、依前款規定計算之薪資所得，於依第十五條規定計算稅額及依第十七條規定計算綜合所\\n得淨額時，不適用第十七條第一項第二款第三目之 2  薪資所得特別扣除之規定。\\n三、第一款各目費用之適用範圍、認列方式、應檢具之證明文件、第二目符合規定之機構、\\n第三目一定金額及攤提折舊或攤銷費用方法、年限及其他相關事項之辦法，由財政部定\\n之。\\n四、第一款薪資收入包括：薪金、俸給、工資、津貼、歲費、獎金、紅利及各種補助費。但\\n為雇主之目的，執行職務而支領之差旅費、日支費及加班費不超過規定標準者，及依第\\n四條規定免稅之項目，不在此限。\\n五、依勞工退休金條例規定自願提繳之退休金或年金保險費，合計在每月工資百分之六範圍\\n內，不計入提繳年度薪資收入課稅；年金保險費部分，不適用第十七條有關保險費扣除\\n之規定。\\n第四類：利息所得：凡公債、公司債、金融債券、各種短期票券、存款及其他貸出款項利息之所\\n得：\\n一、公債包括各級政府發行之債票、庫券、證券及憑券。\\n二、有獎儲蓄之中獎獎金，超過儲蓄額部分，視為存款利息所得。\\n三、短期票券指期限在一年期以內之國庫券、可轉讓銀行定期存單、公司與公營事業機構發\\n行之本票或匯票及其他經目的事業主管機關核准之短期債務憑證。\\n短期票券到期兌償金額超過首次發售價格部分為利息所得，除依第八十八條規定扣繳稅款\\n外，不併計綜合所得總額。\\n第五類：租賃所得及權利金所得：凡以財產出租之租金所得，財產出典典價經運用之所得或專利\\n權、商標權、著作權、秘密方法及各種特許權利，供他人使用而取得之權利金所得：\\n一、財產租賃所得及權利金所得之計算，以全年租賃收入或權利金收入，減除必要損耗及費\\n用後之餘額為所得額。\\n二、設定定期之永佃權及地上權取得之各種所得，視為租賃所得。\\n三、財產出租，收有押金或任何款項類似押金者，或以財產出典而取得典價者，均應就各該\\n款項按當地銀行業通行之一年期存款利率，計算租賃收入。\\n')"
  ]
}
```
