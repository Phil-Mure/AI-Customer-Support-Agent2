import os
import bs4
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnableLambda
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.output_parsers import StrOutputParser


# --- Load environment variables ---
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

# --- Google API Key Setup ---
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- Google gemimi Setup ---
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- LangSmith Tracing ---
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# --- Getting the DB ---
db = SQLDatabase.from_uri("sqlite:///cardTransactions.db")

# --- Initializing Vector storage for documents ---

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# --- Load and chunk contents of the web pages ---
urls = (
    "https://www.infinitepay.io", "https://www.infinitepay.io", 
    "https://www.infinitepay.io/maquininha", "https://www.infinitepay.io/maquininha-celular", 
    "https://www.infinitepay.io/tap-to-pay", "https://www.infinitepay.io/pdv", ""
    "https://www.infinitepay.io/receba-na-hora", 
    "https://www.infinitepay.io/gestao-de-cobranca-2",
    "https://www.infinitepay.io/gestao-de-cobranca", "https://www.infinitepay.io/link-de-pagamento",
    "https://www.infinitepay.io/loja-online", "https://www.infinitepay.io/boleto", 
    "https://www.infinitepay.io/conta-digital", "https://www.infinitepay.io/conta-pj",
    "https://www.infinitepay.io/pix", "https://www.infinitepay.io/pix-parcelado", 
    "https://www.infinitepay.io/emprestimo", "https://www.infinitepay.io/cartao",
    "https://www.infinitepay.io/rendimento"
    )

loader = WebBaseLoader(
    web_paths=urls,
    # bs_kwargs=dict(
    #     parse_only=bs4.SoupStrainer(
    #         class_=("post-content", "post-title", "post-header")
    #     )
    # ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

graph_builder = StateGraph(MessagesState)

# --- Determining Labels for Knowledge Base and general greetings---
def parse_labels(raw_output: str) -> list[str]:
    return [
        line.split(". ", 1)[1].strip()
        for line in raw_output.strip().splitlines()
        if ". " in line
    ]

def label_extractor():
    prompt = PromptTemplate(
        input_variables=["doc"],
        template="""
        Based on the following content, list 5-10 unique and general **support topics** or **FAQs** users might ask. These should serve as intent labels for an AI agent.

        Content:
        {doc}
        """
    )

    label_chain = (
        RunnableMap({"doc": lambda doc: doc.page_content})
        | prompt
        | llm
    )
    return label_chain

def extract_labels(docs):
    label_chain = label_extractor()
    parsed_results = []
    for doc in docs:
        raw = label_chain.invoke(doc).content.strip()
        labels = parse_labels(raw)
        parsed_results.extend(labels)
    return list(set(parsed_results)) 

labels = extract_labels(all_splits[:5])





# Here is a general purpose web search toool
# --- Initialize DuckDuckGo search tool for general purpose websearch ---

search = DuckDuckGoSearchRun()

@tool
def web_search_tool(query: str) -> str:
    """Search the web for recent information using DuckDuckGo."""    
    return search.invoke(query)

# --- Classfy (match) user input with label. This will help determine whether to go ahead with 
# the Knowledge Agent or do the general purpose web seacrh ---
def classify_user_input_with_labels(user_input: str, labels: list[str]) -> str:
    prompt = PromptTemplate.from_template("""
    You are an AI assistant helping match user queries to support topics.

    Here is the user question:
    "{question}"

    Available intent labels:
    {label_list}

    Which one of the labels best matches the user’s question? Respond only with the label text. 
    If no label text is found, answer No Text Found
    """)
    
    chain = prompt | llm
    return chain.invoke({
        "question": user_input,
        "label_list": "\n".join(f"- {label}" for label in labels)
    }).content.strip()


# --- Making the retrieval process as a Tool. --- 
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Here, we leverage another pre-built LangGraph component, 
# ToolNode, that executes the tool and adds the result as a ToolMessage to the state:
# 1. A node that fields the user input, either generating a query for the retriever or responding directly;
# 2. A node for the retriever tool that executes the retrieval step;
# 3. A node that generates the final response using the retrieved context.

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

# --- Implementing the Graph Logic ---
# It should respond appropriately to messages that do not require an additional 
# retrieval step (For example "Hello"):

# I build the logic:
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


# Here is a memory saver that will save the chat state of the graph and specify id of the thread.
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

from langgraph.prebuilt import create_react_agent

# ---This is the Knowledge Agent that will respond to user queries related to knowledge 
# and general purpose websearch.---
def KnowledgeAgent(input_message: str) -> str: 
    """Knowledge Agent that responds to user queries."""

    res = []
    agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
    config = {"configurable": {"thread_id": "def234"}}
    
    input_message = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )
    
    for event in agent_executor.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        res.append(event["messages"][-1].pretty_print())
    return res[0]
    
    # res = "What are the different payment methods?"
    # x = []
    # if classify_user_input_with_labels(input_message, labels) == "No Text Found":
    #     # If no label is found, use the web search tool
    #     response = web_search_tool.invoke(input_message)
    #     res = response
        
    # # If a label is found, use the Knowledge Agent
    # print(f"Knowledge Agent is processing the query: {input_message}")
    # # Stream the graph with the input message
    # # and return the last message
    # # Note: The graph will use the memory saver to save the state of the conversation
    # for step in graph.stream(
    #     {"messages": [{"role": "user", "content": input_message}]},
    #     stream_mode="values",
    #     config=config,
    #     ):
        message = step["messages"][-1].pretty_print()
        x.append(message)
        print(x)
        res = x[0] if res != "None" else res
        print(res)
    return res
    
# --- Creating DB Agent ---
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)


# ---- Customer (SQL) Agent that can answer questions related to the database ---
def CustomerAgent(input_message: str, user_id: int) -> str:
    """Invoke the SQL agent with a specific user_id."""
    prompt = f"{input_message}. Only show results for user_id = '{user_id}'."
    return agent_executor.invoke({"input": prompt})["output"]  # Access the output from the response dictionary

# --- User Labels for Classification ---
user_labels = [
    "transaction inquiry", "dispute charge", "payment failed", "refund request", 
    "number of successful transactions", "mernchert support", "account balance",
    "transaction history", "transaction details", "transaction status",
    "transaction date", "transaction amount", "transaction type", "transaction method",
    "transaction location", "transaction merchant", "transaction description",
    "transaction category", "transaction tags", "transaction notes", "transaction comments"
]


# --- Personality Layer to generate friendly and helpful responses ---
def PersonalityLayer(input_message: str) -> str:
    """Generate a response with a friendly and helpful tone."""
    
    human_prompt_template = PromptTemplate.from_template("""
        You are an intelligent and helpful AI assistant who responds like a friendly, empathetic human. 
        Use a warm, conversational tone and avoid sounding robotic.

        Always:
        - Explain things clearly
        - Use natural language
        - Be concise but friendly

        Here’s the question:
        {input}

        Reply as if you're chatting with the user directly.
    """)
    chain = human_prompt_template | llm | StrOutputParser()
    response = chain.invoke({"input": input_message})
    return response


labels_a = labels  # Knowledge Agent labels
labels_b = user_labels  # Customer Agent labels

router_prompt = PromptTemplate.from_template("""
You are a router deciding whether a user query matches one of two label groups.

Group A (Knowledge Agent): {labels_a}
Group B (Customer Agent): {labels_b}
[EXPLICIT] — if it contains explicit content (e.g. porn, sex, adult content)

Reply with only one of the following labels: A, B, or EXPLICIT.

User input: "{input}"

Decide which group this input belongs to. Only respond with "A" or "B".
""")

# --- Router Chain ---
router_chain = (
    router_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda decision: decision.strip().upper())
)

# --- Async Router Logic ---
def router(user_input: str, user_id: int) -> str:
    decision = router_chain.invoke({
        "input": user_input,
        "labels_a": ", ".join(labels_a),
        "labels_b": ", ".join(labels_b),
    })

    if decision == "A":
        source_agent_response = KnowledgeAgent(user_input)
        response = PersonalityLayer(source_agent_response)
        return {
            "response": response,
            "source_agent_response": source_agent_response,
            "agent_workflow": [{"agent_name": "Knowledge Agent", "tool_calls": {"WebSearchTool": web_search_tool(user_input)}}]
        }
    elif decision == "B":
        source_agent_response = CustomerAgent(user_input, user_id)
        response = PersonalityLayer(source_agent_response)
        return {
            "response": response,
            "source_agent_response": source_agent_response,
            "agent_workflow": [{"agent_name": "Customer Support Agent", "tool_calls": {"SQLSearchTool": "SQL Results", "SQLLabelExtractorTool": user_labels}}]
        }
    else:
        response = "Could not route the query, please clarify. It may contain explicit content or be outside the scope of our agents."
        return {
            "response": response,
            "source_agent_response": "None",
            "agent_workflow": [{"agent_name": "None", "tool_calls": "None"}]
        }

