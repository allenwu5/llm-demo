from langchain_openai import ChatOpenAI

from src.config import *

llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
