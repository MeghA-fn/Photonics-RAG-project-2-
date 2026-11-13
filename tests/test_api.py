
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Photonics RAG API is running!"
    }


def test_ask_returns_no_answer_when_no_documents(monkeypatch):

    # Mock retrieval to return no documents
    monkeypatch.setattr(
        "app.api.hybrid_search",
        lambda question: []
    )

    response = client.post(
        "/ask",
        json={"question": "What is stimulated emission?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "no_answer_found"
    assert data["confidence"] == 0.0
    assert data["sources"] == []


def test_ask_returns_generated_answer(monkeypatch):

    # Mock retrieved documents
    monkeypatch.setattr(
        "app.api.hybrid_search",
        lambda question: [
            {
               "text": "Stimulated emission occurs when an incoming photon triggers an excited atom to emit another photon.",
               "document_name": "laser_physics.pdf",
               "page_number": 3,
               "distance": 0.2
            }
        ]
    )

    # Mock prompt builder
    monkeypatch.setattr(
        "app.api.build_prompt",
        lambda question, documents: "Mock photonics prompt"
    )

    # Mock Gemini answer
    monkeypatch.setattr(
        "app.api.generate_answer",
        lambda prompt: "Stimulated emission produces a photon identical to the incoming photon."
    )

    response = client.post(
        "/ask",
        json={"question": "What is stimulated emission?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] != "no_answer_found"
    assert "stimulated emission" in data["answer"].lower()
    assert len(data["sources"]) > 0
