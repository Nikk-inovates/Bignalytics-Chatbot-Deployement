from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise ValueError("Missing API_TOKEN in environment")

API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/47055a3a6ea4207316ce59689541eae5/ai/run/"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def run(model, inputs):
    try:
        response = requests.post(f"{API_BASE_URL}{model}", headers=HEADERS, json={"messages": inputs})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"response": f"API call failed: {e}"}

# Load FAISS index
def load_faiss_index(path="faiss_index"):
    global retriever
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="mmr", k=12)
    return retriever

# Get top-k context
def get_top_k_context(question):
    return retriever.invoke(question)

# Generate response from context
def generate_response(context, question, use_api=False):
    prompt_template = """
You are a helpful chatbot assistant for Bignalytics Institute.
Answer the question using only the context below. Just generate output from the given context.
If it is not present in context, don't make it up and just say "I am sorry can you ask another question".

Context:
{context}

Question:
{question}
"""
    full_prompt = prompt_template.format(context=context, question=question)

    if use_api:
        inputs = [
            {"role": "system", "content": "You are a helpful chatbot assistant for Bignalytics Institute."},
            {"role": "user", "content": full_prompt}
        ]
        response = run("@cf/meta/llama-3-8b-instruct", inputs)
        return response.get("response") or response.get("result") or "Sorry, no valid response received from the model."
    else:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        model = OllamaLLM(model="llama3.1:8b")
        rag_chain = (
            {"context": lambda x: x["context"], "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        return rag_chain.invoke({"context": context, "question": question})
