"""
Integration tests for the BBC News Classifier API endpoints.
Covers both the default (tfidf) and explicit model_type override (bert).
"""
import pytest


# ── Health & Info ─────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/news/health")
        assert response.status_code == 200

    def test_health_lists_loaded_models(self, client):
        data = client.get("/api/v1/news/health").json()
        assert data["status"] == "healthy"
        assert "tfidf" in data["loaded_models"]
        assert "bert" in data["loaded_models"]


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        assert client.get("/api/v1/news/model/info").status_code == 200

    def test_model_info_has_categories(self, client):
        data = client.get("/api/v1/news/model/info").json()
        cats = data["supported_categories"]
        assert set(cats) == {"sport", "business", "politics", "tech", "entertainment"}

    def test_model_info_has_active_models(self, client):
        data = client.get("/api/v1/news/model/info").json()
        assert isinstance(data["active_models"], list)


# ── Single Classification ─────────────────────────────────────────────────

class TestClassifyEndpoint:
    def test_classify_success_tfidf(self, client):
        payload = {"content": "Arsenal beat Tottenham 3-0 in the north London derby."}
        response = client.post("/api/v1/news/classify?model_type=tfidf", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["result"]["label"] in {"sport", "business", "politics", "tech", "entertainment"}
        assert data["result"]["model_used"] == "tfidf"

    def test_classify_success_bert(self, client):
        payload = {"content": "Arsenal beat Tottenham 3-0 in the north London derby."}
        response = client.post("/api/v1/news/classify?model_type=bert", json=payload)
        assert response.status_code == 200
        assert response.json()["result"]["model_used"] == "bert"

    def test_classify_with_headline(self, client):
        payload = {
            "headline": "Markets rise after Fed announcement",
            "content": "Wall Street indexes climbed sharply on Wednesday...",
        }
        response = client.post("/api/v1/news/classify", json=payload)
        assert response.status_code == 200

    def test_classify_empty_content_422(self, client):
        response = client.post("/api/v1/news/classify", json={"content": ""})
        assert response.status_code == 422

    def test_classify_missing_content_422(self, client):
        response = client.post("/api/v1/news/classify", json={"headline": "No body"})
        assert response.status_code == 422

    def test_classify_content_too_long_422(self, client):
        response = client.post(
            "/api/v1/news/classify",
            json={"content": "a" * 10_001},
        )
        assert response.status_code == 422

    def test_classify_invalid_model_type_422(self, client):
        payload = {"content": "Some news story.", "model_type": "invalid_model"}
        response = client.post("/api/v1/news/classify", json=payload)
        assert response.status_code == 422


# ── Batch Classification ──────────────────────────────────────────────────

class TestBatchClassifyEndpoint:
    def test_batch_classify_success(self, client):
        payload = {
            "articles": [
                {"content": "Arsenal beat Tottenham 3-0 in the derby."},
                {"content": "The Chancellor announced a new budget today."},
            ]
        }
        response = client.post("/api/v1/news/classify/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_batch_with_bert_override(self, client):
        payload = {
            "articles": [{"content": "Silicon Valley startup raises $1bn."}],
            "model_type": "bert",
        }
        response = client.post("/api/v1/news/classify/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["results"][0]["model_used"] == "bert"

    def test_batch_empty_articles_422(self, client):
        response = client.post("/api/v1/news/classify/batch", json={"articles": []})
        assert response.status_code == 422

    def test_batch_exceeds_max_422(self, client):
        articles = [{"content": f"Article number {i} content."}  for i in range(33)]
        response = client.post("/api/v1/news/classify/batch", json={"articles": articles})
        assert response.status_code == 422


# ── Root redirect ─────────────────────────────────────────────────────────

class TestRootRedirect:
    def test_root_redirects_to_docs(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (301, 302, 307)
        assert response.headers["location"] == "/docs"
