from pdf_preprocessor import load_pdf_documents, load_embedding_model, create_semantic_vector_store

# Define path
PDF_PATH = r"E:\INTERNSHIP\app\data\Dataset.pdf"

# Load and preprocess
docs = load_pdf_documents(PDF_PATH)
embeddings = load_embedding_model()

# Build and save vector store
create_semantic_vector_store(docs, embeddings)

print("✅ FAISS index built and saved.")
