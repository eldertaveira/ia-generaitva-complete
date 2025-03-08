import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def process_files(files):
    """
    Processa arquivos PDF, CSV e Excel. Retorna:
    - 'text_data': Texto extraído dos PDFs.
    - 'table_data': Dicionário contendo DataFrames de arquivos CSV e Excel.
    """
    text_data = ""
    table_data = {}

    for file in files:
        filename = file.name.lower()

        if filename.endswith('.pdf'):
            pdf = PdfReader(file)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_data += page_text + "\n"

        elif filename.endswith('.csv'):
            df = pd.read_csv(file)
            table_data[filename] = df  # Armazena o DataFrame

        elif filename.endswith('.xls') or filename.endswith('.xlsx'):
            df = pd.read_excel(file)
            table_data[filename] = df  # Armazena o DataFrame

    return text_data, table_data


def create_text_chunks(text):
    """
    Divide o texto em partes menores para melhor processamento do chatbot.
    """
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=8000,
        chunk_overlap=1000,
        length_function=len
    )
    return text_splitter.split_text(text)
