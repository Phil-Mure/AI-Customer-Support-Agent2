from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict
from agents import router


# --- Initialize FastAPI ---
app = FastAPI(
    title="Customer Support AI Agent API", 
    description="""Routes user input to Knowledge or Customer Agent using LangChain, LangGraph, OpenAI 
                    and Google Gemini. The Knowledge Agent provides information based on indexed web content from support pages and FQAs. 
                    It searches the web if no information is found on the pages. The Customer Agent handles account-related queries using 
                    an SQL database. The API uses a router to determine which Agent to invoke based on user input. The final layer is the 
                    Personality layer (workfow) that helps to make it more like a human being responding. The user_id should be an 
                    integer from 1 to 100.
                """, 
    version="1.0"
    )

# --- Pydantic models ---
class QueryRequest(BaseModel):
    user_id: int
    input: str

@app.post("/route")
def post_route(request: QueryRequest):
    res = router(user_input=request.input, user_id=request.user_id)
    return res
