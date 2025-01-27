from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI 
from dotenv import load_dotenv
import os
load_dotenv()
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)
    return vectorstore

def setup_qa_chain(vectorstore):
    api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(openai_api_key = api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain
