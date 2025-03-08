from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


# Função process_files recebe os documentos carregados realiza a extração do texto do PDFs
def process_files(files):

    text = ""

    for file in files:
        pdf = PdfReader(file)

        # iterar por cada pagina e  pegar todo o texto e transforma em string
        for page in pdf.pages:
            text += page.extract_text()
    
    return text


# a Função create_text_chunks particiona o texto extraido em textos menores
def create_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = '\n',       # Caractere usado para dividir os textos
        chunk_size = 8000,      # Descrevem o tamanho máximo que um único pedaço de chunk pode ter
        chunk_overlap = 1000,    # Quantidade de caracteres que sobrepos um chunck de outro
        length_function = len
    )
    chunks = text_splitter.split_text(text)

    return chunks

