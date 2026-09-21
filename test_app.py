"""
Integration and Unit Tests for Moodline FastAPI Emotion Analysis Server
"""
import sys
from fastapi.testclient import TestClient
from main import app

def run_tests():
    print("Starting Moodline test suite...")
    with TestClient(app) as client:
        # 1. Health Check
        print("Test 1: Testing /health endpoint...")
        response = client.get("/health")
        assert response.status_code == 200, f"Health check failed: {response.text}"
        data = response.json()
        assert data["status"] == "healthy", f"Status not healthy: {data}"
        assert data["model_loaded"] is True, "Model not reported loaded"
        assert len(data["classes"]) == 6, "Expected 6 emotion classes"
        print("  -> Passed! Model loaded and classes verified.")

        # 2. Prediction - Joy
        print("Test 2: Testing /predict with joyful sentence...")
        joy_text = "I feel so completely delighted, thrilled and happy today!"
        response = client.post("/predict", json={"text": joy_text})
        assert response.status_code == 200, f"Prediction failed: {response.text}"
        data = response.json()
        assert data["predicted_emotion"] in ["joy", "love"], f"Unexpected emotion: {data['predicted_emotion']}"
        assert data["confidence"] > 0.5, f"Low confidence: {data['confidence']}"
        assert "all_probabilities" in data
        assert "all_probabilites" in data  # Backwards compatibility check
        assert len(data["all_probabilities"]) == 6
        assert data["latency_ms"] >= 0
        print(f"  -> Passed! Predicted: {data['predicted_emotion']} ({data['confidence']*100:.1f}%) in {data['latency_ms']}ms")

        # 3. Prediction - Sadness
        print("Test 3: Testing /predict with sad sentence...")
        sad_text = "I feel lonely, miserable, and heartbroken today."
        response = client.post("/predict", json={"text": sad_text})
        assert response.status_code == 200, f"Prediction failed: {response.text}"
        data = response.json()
        assert data["predicted_emotion"] in ["sadness", "fear"], f"Unexpected emotion: {data['predicted_emotion']}"
        assert data["confidence"] > 0.5
        print(f"  -> Passed! Predicted: {data['predicted_emotion']} ({data['confidence']*100:.1f}%) in {data['latency_ms']}ms")

        # 4. Validation Rejection - Whitespace or Empty
        print("Test 4: Testing validation handling for blank text...")
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code in [422, 400], f"Expected validation error, got {response.status_code}"
        print("  -> Passed! Invalid input cleanly rejected.")

        # 5. UI Endpoint
        print("Test 5: Testing GET / endpoint...")
        response = client.get("/")
        assert response.status_code == 200, f"UI route failed: {response.status_code}"
        assert "text/html" in response.headers.get("content-type", "")
        print("  -> Passed! Static index.html served successfully.")

    print("\nAll 5 automated tests passed with flying colors! [SUCCESS]")

if __name__ == "__main__":
    run_tests()
