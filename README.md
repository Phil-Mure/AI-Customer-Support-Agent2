# 🧠 LangChain-Gemini Fintech AI Customer Support Agent API

This is a FastAPI-based AI backend that uses **LangChain** and **Google Gemini** to process natural language input related to **card transactions** and route the request to the correct AI agent (Knowledge Agent or Customer Agent). It connects to a `SQLite` database (`cardTransactions.db`) and supports JSON-based POST queries. The Knowledge Agent provides information based on indexed web content from support pages and FQAs. It searches the web if no information is found on the pages. The Customer Agent handles account-related queries using an SQL database. The API uses a router to determine which Agent to invoke based on user input. The final layer is the Personality layer (workfow) that helps to make it more like a human being responding. The user_id should be an integer from 1 to 100.

---

## 🚀 Features

* 🌐 FastAPI server with RESTful endpoint (`/route`)
* 🔮 Google Gemini + LangChain for natural language processing
* 🧠 Agent routing logic based on input labels (e.g., `Card Activation`, `Refund`, etc.)
* 🐬 SQLite database support
* 🐼 Pandas (Extracting CSV data and creating SQL Tables)
* 🧪 Pytest test suite for endpoint validation
* 📦 Dockerized deployment with `docker-compose`
* 🌱 LangSmith integration for debugging and trace visualization
* 🔐 Environment configuration via `.env`

---

## 📂 Project Structure

```
.
├── agents.py                # LangChain, Gemini and Pandas logic
├── main.py                  # FastAPI app logic
├── cardTransactions.db      # Sample SQLite database
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container build config
├── docker-compose.yml       # Multi-container setup
├── .env                     # Environment variables
├── test_main.py             # Pytest tests for endpoint
└── README.md                # You're reading this
```

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Phil-Mure/AI-Customer-Support-Agent2.git
cd AI-Customer-Support-Agent2
```

### 2. Set Up Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your-google-api-key
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=fintech-ai
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use Docker:

```bash
docker-compose up --build
```

---

## 🧠 API Usage

### POST `/route`

**Request:**

```json
{
  "user_id": 1,
  "user_input": "What's the total number of successful transactions in my account?"
}
```

**Response:**

```json
{
  "response": "Hi, I'm glad to help you with your question. Kindly note that the total number of successful transactions in your account is 8.",
  "source_agent_response": "The total number of successful transactions for this user of id 1 is 8",
  "agent_workflow": [{"agent_name": "Customer Agent", "tool_calls": {"SQLSearchTool": "SQL Results", "SQLLabelExtractorTool": "SQL Labels"}}]
}
```

---

## 🦪 Running Tests

Run unit tests using:

```bash
pytest test_main.py
```

---

## 🐳 Docker Deployment

### Build and Start the App

```bash
docker-compose up --build
```

### Access FastAPI Docs

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔍 LangSmith Debugging

Track and visualize your LangChain chains in [LangSmith](https://smith.langchain.com):

* Set `LANGCHAIN_API_KEY` in `.env`
* Your runs will appear automatically

---

## 🧠 Agent Labels Used

Used for routing user input. 
1. *Customer Agent Labels*:

* `Card Activation`
* `Card Blocking`
* `Payment Methods`
* `Refunds & Chargebacks`
* `Security & Fraud`
* `Transaction History`
* `Fees & Pricing`
* `Total Number of Transactions`
* `Total Merchants`
* ...and more (see `agents.py`)

2. *Knowledge Agent Labels*:
This is a list of **Support Topics** or **FAQ** extracted from the indexed documents.
* ...see `extract_labels` function in `agents.py`

---

## 🧠 Tech Stack

* 🐍 Python 3.10+
* 🐼 Pandas
* ⚡ FastAPI
* 🔮 LangChain + Gemini 2.0 Flash
* 🧠 LangSmith
* 🐬 SQLite
* 🐳 Docker + Docker Compose
* ✅ Pytest

---

## 📌 Notes

* The SQLite DB must exist (`cardTransactions.db`) or be generated before querying.
* This project is designed to be easily extended to other LLMs and vector stores.

---

## 📃 License

MIT License — free to use and modify.

---

## 🤛‍♂️ Support

Need help or want to contribute?
Open an issue or reach out at `pmurebwa@gmail.com`.


