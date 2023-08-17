import streamlit as st
import os, sys, time
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.subheader("Doc-Bot 🤖 ")
st.write(OPENAI_API_KEY)

chat_history = []

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    @st.cache_resource
    def get_documents():
        documents = []
        # Create a List of Documents from all of our files in the ./docs folder
        for file in os.listdir("docs"):
            if file.endswith(".pdf"):
                pdf_path = "./docs/" + file
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                doc_path = "./docs/" + file
                loader = Docx2txtLoader(doc_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                text_path = "./docs/" + file
                loader = TextLoader(text_path)
                documents.extend(loader.load())

        return documents
    
    @st.cache_resource
    def pdf_chain(_documents):
        # Split the documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        docs = text_splitter.split_documents(_documents)

        # Convert the document chunks to embedding and save them to the vector store
        vectordb = Chroma.from_documents(docs, embedding=OpenAIEmbeddings(), persist_directory="./data")
        vectordb.persist()

        # create our Q&A chain
        pdf_qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
            retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
            return_source_documents=True,
            verbose=True
        )
        return pdf_qa
    

    if "btn_state" not in st.session_state:
        st.session_state.btn_state = False

    btn = st.button("Initialize Bot")

    if btn or st.session_state.btn_state:
        st.session_state.btn_state = True
        documents = get_documents()
        pdf_qa = pdf_chain(documents)
        st.success("Bot Ready ☑️! ")

        if "messages" not in st.session_state:
            st.session_state.messages=[]

        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        if prompt := st.chat_input("What's up?"):
            st.session_state.messages.append({"role":"user", "content":prompt})
            print(st.session_state)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Collecting answer"):
                    message_placeholder = st.empty()
                    full_response = ""
                    result = pdf_qa({"question":prompt,"chat_history":chat_history})
                    assistant_response = result['answer']
                    chat_history.append((prompt,assistant_response))

            for chunk in assistant_response:
                full_response += chunk + ""
                time.sleep(0.01)
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role":"assistant", "content":full_response})


    else:
        st.info('Click on button below to initialize bot')
            

            



        
        