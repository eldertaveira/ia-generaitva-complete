import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not api_key:
    raise ValueError(
        "A chave da API do Gemini não foi encontrada. Verifique o arquivo .env ou Streamlit Secrets.")

if google_credentials and not os.path.exists(google_credentials):
    raise ValueError(
        f"O arquivo de credenciais do Google não foi encontrado: {google_credentials}")

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=api_key)


def create_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document",
        google_api_key=api_key
    )

    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    if vectorstore:
        print("Vectorstore criado com sucesso!")
    else:
        print("Erro ao criar o Vectorstore!")

    return vectorstore


def create_conversation_chain(vectorstore):
    model_Gemini = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=api_key
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        model_Gemini,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


def analyze_table(df, question):
    """
    Usa um modelo de IA para responder perguntas sobre DataFrames.
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=api_key)  # ✅ Modelo corrigido!

    table_preview = df.head().to_string()  # Mostra as primeiras linhas da tabela
    prompt = f"A tabela abaixo contém dados estruturados. Responda à seguinte pergunta com base nesses dados:\n\n{table_preview}\n\nPergunta: {question}"

    response = model.invoke(prompt)

    return response["content"] if isinstance(response, dict) and "content" in response else str(response)
