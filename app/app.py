from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from ragpipeline import load_faiss_index, get_top_k_context, generate_response
from feedback import save_feedback_txt
import uvicorn

import os
app = FastAPI()

# CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
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
    response = generate_response(context=context, question=query.question,use_api=True)

    # Flexible handling of response format
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



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
