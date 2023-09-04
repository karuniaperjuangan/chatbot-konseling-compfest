"""
Ini adalah kode utama untuk menjalankan aplikasi chatbot
"""

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import faiss
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
from langchain.vectorstores import FAISS
from typing import List
import gradio as gr
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0)

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store:FAISS = pickle.load(f)

store.index = index

def ask_retrieval_qa(question:str,history:List[List[str]]):
    answers = store.similarity_search("Apa kontak yang bisa dihubungi untuk menghadapi kasus kekerasan seksual?", k=1)
    messages = [
        SystemMessage(content=f"""Kamu adalah sebuah chatbot yang bernama Bangkit. Tugasmu adalah membantu orang-orang yang mengalami masalah kekerasan seksual. Kamu akan memberikan informasi yang dapat membantu seseorang untuk melakukan tindak lanjut terhadap kasus kekerasan seksual yang dialami.
Anda bisa menggunakan potongan UU No 12 tahun 2022 berikut untuk membantu Anda:
{answers[0].page_content}
Sumber: {answers[0].metadata}"""),
        
    ]
    for human,ai in history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    messages.append(HumanMessage(content=question))
    response = chat(messages)
    return response.content

with gr.Blocks() as demo:
    
    gr.Markdown("""<h1 style="text-align:center">Aplikasi Bangkit (Intelligent Chatbot for Sexual Violence Behaviour) </h1>""")
    gr.Markdown(
        """
        Gunakan chatbot di bawah untuk menangani kasus kekerasan seksual yang anda alami.
        """        )
    gr.ChatInterface(ask_retrieval_qa)
demo.launch(auth=(os.environ.get("GRADIO_USERNAME"),os.environ.get("GRADIO_PASSWORD")))

