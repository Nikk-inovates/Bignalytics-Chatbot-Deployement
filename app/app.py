import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from app.ragpipeline import load_faiss_index, get_top_k_context, generate_response
from app.feedback import save_feedback_txt
from dotenv import load_dotenv  # ✅ Load environment variables

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Read PORT from environment (fallback to 8000)
PORT = int(os.getenv("PORT", 8000))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://zingy-tapioca-5cd938.netlify.app")

app = FastAPI()

# ✅ Secure CORS settings for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_faiss_index()

class QueryInput(BaseModel):
    question: str

class QueryOutput(BaseModel):
    context: str
    response: str

@app.post("/ask", response_model=QueryOutput)
def ask_question(query: QueryInput):
    context_docs = get_top_k_context(query.question)
    context = "\n".join(doc.page_content for doc in context_docs)
    response = generate_response(context=context, question=query.question, use_api=True)

    final_response = response["response"] if isinstance(response, dict) and "response" in response else response

    return {
        "context": context,
        "response": final_response
    }

class FeedbackInput(BaseModel):
    question: str
    context: str
    response: str
    feedback: str

@app.post("/feedback")
def submit_feedback(data: FeedbackInput):
    save_feedback_txt(data.question, data.context, data.response, data.feedback)
    return {"status": "success", "message": "Feedback saved."}

@app.get("/download-feedback")
def download_feedback():
    file_path = "feedback_logs.csv"
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename="feedback_logs.csv",
            media_type="text/csv"
        )
    else:
        return {"error": "Feedback log file does not exist."}

# ✅ Production-safe entry point (no reload, dynamic port)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT)
