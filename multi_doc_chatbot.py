import os, sys, time
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import VectorStoreRetrieverMemory
load_dotenv()

def add_filename_to_file(filename, txtfile_path):
    with open(txtfile_path,'a') as f:
        f.write(filename+"\n")

def inTxtfile(filename,txtfile_path):
    with open(txtfile_path,'r') as f:
        return filename in f.read().splitlines()
    
# EMBEDDED_FILES_PATH = os.path.join('EMBEDDED_FILE_NAMES.txt')

def get_documents():
    docs = []
    s = time.time()
    # Create a List of Documents from all of our files in the ./docs folder
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            pdf_path = "./docs/" + file
            loader = PyPDFLoader(pdf_path) 
            docs.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = "./docs/" + file
            loader = Docx2txtLoader(doc_path)
            docs.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = "./docs/" + file
            loader = TextLoader(text_path)
            docs.extend(loader.load())
        elif file.endswith('.csv'):
            csv_path = './docs/' + file
            loader=CSVLoader(csv_path,'en',delimiter=',')
            docs.extend(loader.load())
    print(time.time()-s)

    return docs

documents = get_documents()

def pdf_chain(docs):
    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(docs)

    # Convert the document chunks to embedding and save them to the vector store
    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data",)
    vectordb.persist()

    # create our Q&A chain
    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=True
    )

    return pdf_qa

pdf_qa = pdf_chain(documents)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa({"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))