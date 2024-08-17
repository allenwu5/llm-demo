from src.config import *
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")