from fastapi.testclient import TestClient
from main import app  


# This is a test file for the FastAPI application that routes user queries to different agents.
# It tests the routing logic to ensure that the correct agent is invoked based on user input and that 
# the responses are as expected.

# --- Initialize FastAPI TestClient ---
client = TestClient(app)

def test_knowledge_agent_routing():
    payload = {
        "user_id": 1,
        "user_input": "How much is the infinite card charges?"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(data)
    # Check if the response contains the expected keys and values
    assert "response" in data
    assert "source_agent_response" in data
    assert data["agent_workflow"]["agent_name"] == "Knowledge Agent"
    # assert data["agent_workflow"]["tool_calls"]["WebSearchTool"] == "Web Search"

def test_customer_agent_routing():
    payload = {
        "user_id": 2,
        "user_input": "What's the total number of transactions in my acoount?"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(data)
    # Check if the response contains the expected keys and values
    assert "response" in data
    assert "source_agent_response" in data
    assert data["agent_workflow"]["agent_name"] == "Customer Agent"
    # assert data["agent_workflow"]["tool_calls"]["SQLLabelExtractorTool"] == "SQL Labels"
    # assert data["agent_workflow"]["tool_calls"]["SQLSearchTool"] == "SQL Results"
