import streamlit as st
from utils import chatbot, text
from streamlit_chat import message


def main():

    st.set_page_config(page_title='Chat com Arquivos', page_icon=':books:')
    st.header('Chat com PDFs e Tabelas')

    # Inicializa variáveis de sessão
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'table_data' not in st.session_state:
        st.session_state.table_data = None

    user_question = st.text_input(
        "Pergunte algo sobre os arquivos", placeholder="Escreva aqui...")

    if user_question:
        # Se houver PDFs processados, usamos a busca semântica com FAISS
        if st.session_state.conversation:
            response = st.session_state.conversation(
                {'question': user_question})
            if 'chat_history' in response:
                for i, text_message in enumerate(response['chat_history']):
                    message(text_message.content, is_user=(
                        i % 2 == 0), key=f"{i}_msg")

        # Se houver tabelas carregadas, analisamos com o LLM
        elif st.session_state.table_data:
            selected_table = st.selectbox(
                "Escolha a tabela para análise", list(st.session_state.table_data.keys()))
            df = st.session_state.table_data[selected_table]

            response = chatbot.analyze_table(df, user_question)
            # Extrai apenas o conteúdo relevante da resposta
            if isinstance(response, str):  
                resposta_final = response  # Se já for string, usa diretamente
            elif isinstance(response, dict):  
                resposta_final = response.get("content", "Não foi possível obter uma resposta.")
            elif hasattr(response, "content"):  
                resposta_final = getattr(response, "content", "Não foi possível obter uma resposta.")
            else:  
                resposta_final = str(response)  # Converte qualquer outro tipo para string

            # Agora garantimos que só o texto será exibido, sem metadados extras
            st.write("Resposta:")
            st.write(resposta_final)


    with st.sidebar:
        st.subheader('Seus Arquivos')

        pdf_docs = st.file_uploader(
            "Carregue seus arquivos (PDF, CSV, Excel)",
            type=["pdf", "csv", "xls", "xlsx"],
            accept_multiple_files=True
        )

        if st.button('PROCESSAR') and pdf_docs:
            try:
                text_data, table_data = text.process_files(pdf_docs)

                # Se houver PDFs, processamos para FAISS
                if text_data:
                    chunks = text.create_text_chunks(text_data)
                    vectorstore = chatbot.create_vectorstore(chunks)
                    st.session_state.conversation = chatbot.create_conversation_chain(
                        vectorstore)

                # Se houver tabelas, armazenamos
                if table_data:
                    st.session_state.table_data = table_data

                st.success("Processamento concluído! Faça sua pergunta.")

            except Exception as e:
                st.error(f"Erro ao processar arquivos: {str(e)}")

        # 🔴 BOTÃO DE REINICIAR 🔴 (Agora na parte inferior da tela)
        st.markdown("---")  # Linha separadora para organizar a sidebar
        if st.button("🔄 Reiniciar Aplicação"):
            st.session_state.conversation = None
            st.session_state.table_data = None
            st.rerun()  # Usa a versão mais recente do Streamlit para recarregar a aplicação


if __name__ == '__main__':
    main()
