from langchain_community.document_loaders import DirectoryLoader, CSVLoader, UnstructuredWordDocumentLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, util


def prepare_and_split_docs(directory):
    # Load the documents
    loaders = [
        DirectoryLoader(directory, glob="**/*.pdf",show_progress=True, loader_cls=PyPDFLoader),
        DirectoryLoader(directory, glob="**/*.docx",show_progress=True),
        DirectoryLoader(directory, glob="**/*.csv",loader_cls=CSVLoader)
    ]


    documents=[]
    for loader in loaders:
        data =loader.load()
        documents.extend(data)

    # Initialize a text splitter
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
        disallowed_special=(),
        separators=["\n\n", "\n", " "]
    )

    # Split the documents and keep metadata
    split_docs = splitter.split_documents(documents)

    return split_docs


def ingest_into_vectordb(split_docs, embeddings):
    db = FAISS.from_documents(split_docs, embeddings)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    print("Documents are inserted into FAISS vectorstore")
    return db
