from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
# import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
import os
import json
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

# # Carregar chave da API-key do Gemini e Google_credentials
# api_key = os.getenv("GEMINI_API_KEY")
# google_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# # Verificar se a chave foi carregada corretamente
# if not api_key:
#     raise ValueError(
#         "A chave da API do Gemini não foi encontrada. Verifique o arquivo .env.")
# if not google_credentials:
#     raise ValueError(
#         "As credenciais do Google não foram encontradas. Verifique a variável GOOGLE_APPLICATION_CREDENTIALS no arquivo .env.")

# # Verificar se o arquivo JSON de credenciais do Google existe
# if not os.path.exists(google_credentials):
#     raise ValueError(
#         f"O arquivo de credenciais do Google não foi encontrado no caminho especificado: {google_credentials}")

# # configuração da chave API (api_key) para consumir modelos como o Gemini 1.5 (da Google DeepMind).
# # genai.configure(api_key=api_key)
# llm = GoogleGenerativeAI(model="gemini-1.5", google_api_key=api_key)

# Carregar chave da API-key do Gemini
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

# Configuração do Google Credentials
google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Se estiver rodando no Streamlit Cloud, pegamos do `st.secrets`
if not google_credentials and "google_credentials" in st.secrets:
    # Converte para dicionário
    credentials_dict = dict(st.secrets["google_credentials"])

    # Criar um arquivo temporário para armazenar as credenciais no ambiente online
    credentials_path = "/tmp/credentials.json"
    with open(credentials_path, "w") as f:
        json.dump(credentials_dict, f)

    # Definir a variável de ambiente para apontar para esse arquivo temporário
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    google_credentials = credentials_path

# Verificar se a chave da API foi carregada corretamente
if not api_key:
    raise ValueError(
        "A chave da API do Gemini não foi encontrada. Verifique o arquivo .env ou Streamlit Secrets.")

# Verificar se o arquivo JSON de credenciais do Google existe
if not os.path.exists(google_credentials):
    raise ValueError(
        f"O arquivo de credenciais do Google não foi encontrado no caminho especificado: {google_credentials}")

# Configurar o modelo Gemini
llm = GoogleGenerativeAI(model="gemini-1.5", google_api_key=api_key)


def create_vectorstore(chunks):
    # Inicializa o modelo de embeddings Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        # Esse modelo é otimizado para gerar embeddings semânticos.
        model="models/embedding-001",
        task_type="retrieval_document",
        google_api_key=api_key  # linha adicionada 27/02
    )

    # GoogleGenerativeAIEmbeddings → Usa o modelo Gemini para gerar embeddings.
    # FAISS → Cria o banco vetorial para fazer busca semântica.
    # from_texts → Converte os textos em vetores e armazena no FAISS.
    # Cria o banco vetorial para fazer busca semântica.
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    print("Vectorstore criado com sucesso!")

    return vectorstore


def create_conversation_chain(vectorstore):
    model_Gemini = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=api_key  # linha adicionada 27/02
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        model_Gemini,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
