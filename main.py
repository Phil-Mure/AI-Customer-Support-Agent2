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
                    and Google Gemini. The Knowledge Agent provides information based on indexed web 
                    content from support pages and FQAs. It searches the web if no information is found on the pages.
                    the Customer Agent handles account-related queries using an SQL database. The API uses a router to determine 
                    which Agent to invoke based on user input. The user_id should an integer 
                    from 1 to 100.
                """, 
    version="1.0"
    )

# --- Pydantic models ---
class QueryRequest(BaseModel):
    user_id: int
    input: str

class QueryResponse(BaseModel):
    response: str
    source_agent_response: str
    agent_workflow: List
    agent_name: str
    tool_calls: Dict

@app.get("/route", response_model=QueryResponse)
def get_route(user_id: int, input: str):
    return router(user_id, input)

@app.post("/route", response_model=QueryResponse)
def post_route(request: QueryRequest):
    return router(request.input)
