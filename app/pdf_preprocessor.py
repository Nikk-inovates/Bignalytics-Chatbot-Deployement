from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load PDF Document
def load_pdf_documents(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()

# 2. Load Embedding Model
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

# 3. Create Semantic Vector Store (FAISS)
def create_semantic_vector_store(docs, embeddings=load_embedding_model()):
    semantic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # can adjust between 800–1500
    chunk_overlap=300,
    separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = semantic_splitter.split_documents(docs)
    # print(split_docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store.save_local("faiss_index")



