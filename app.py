import streamlit as st

from utils import chatbot, text
from streamlit_chat import message

def main():
    
    st.set_page_config(
        page_title='ChatPDF TRE-RN', 
        page_icon=':books:'
    )

    st.header('Chat PDF TRE-RN')
    user_question = st.text_input("Pergunte alguma coisa", placeholder="Escreva aqui...")

    if('conversation' not in st.session_state): # Armazen as variáveis dentro de um estado de sessão
        st.session_state.conversation = None
    
     # Inserindo as mensagens do usario dentro da nossa interface
    if(user_question):
        response = st.session_state.conversation(user_question) ['chat_history']

        for i, text_message in enumerate(response):

            if(i % 2 == 0):
                message(text_message.content, is_user=True, key=str(i) + '_user')
            else:
                message(text_message.content, is_user=False, key=str(i) + '_bot')

    # adicionar interatividade ao seu aplicativo e organizá em uma barra lateral. 
    with st.sidebar:
        st.subheader('Seus Arquivos')

        # accept_multiple_files=True, permite que o usuário carregue vários arquivos ao mesmo tempo.
        # st.file_uploader - Exibir um widget de upload de arquivo carregados limitados a 200 MB. 
        pdf_docs = st.file_uploader("Carregue seus arquivos em formato PDF aqui !", accept_multiple_files=True)

        if st.button('PROCESSAR'):

            # Chama a função process_files e passa pra ela os documentos carregados pelo file_uploader
            all_files_text = text.process_files(pdf_docs)
            
            #Chama função create_text_chunks com os textos dos documentos e recebe os chunks
            chunks = text.create_text_chunks(all_files_text)

            #criando o Vectorstore para armazenar os vetores de embeddings das partes do texto (chunks)
            vectorstore = chatbot.create_vectorstore(chunks)

            # Cria e armazena uma conversa (conversation) na sessão do Streamlit usando o chatbot.
            st.session_state.conversation = chatbot.create_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()