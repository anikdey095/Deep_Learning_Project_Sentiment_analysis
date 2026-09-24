import os
import sys
import time
import httpx

# Ensure proper UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

LIVE_URL = "https://moodline-sentiment.onrender.com"

sentences = [
    "I feel so alone and hopeless today.",
    "I love my family and friends with all my heart.",
    "I am furious and outraged by this terrible service!",
    "A cold shiver crept down my spine as darkness engulfed the deserted alley.",
    "I was shocked and completely surprised by the unexpected gift!",
    "Hello world",
    "This is a test",
    "How are you",
    "I feel good",
    "I feel bad",
    "Good morning"
]

print(f"Connecting to live instance: {LIVE_URL} ...")
client = httpx.Client(base_url=LIVE_URL, timeout=45.0)

# Check health first
try:
    health_resp = client.get("/health")
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        print(f"[OK] Health Check Passed: Model={health_data.get('model_name')}, Ready={health_data.get('model_loaded')}, Uptime={health_data.get('uptime_seconds')}s\n")
    else:
        print(f"[WARN] Health endpoint returned HTTP {health_resp.status_code}\n")
except Exception as e:
    print(f"[FAIL] Health check failed: {e}\n")

print(f"{'Input Sentence':<42} -> {'Emotion':<10} {'Confidence':<10} {'Latency':<10}")
print("-" * 75)

for s in sentences:
    start = time.perf_counter()
    try:
        r = client.post("/predict", json={"text": s})
        elapsed_ms = (time.perf_counter() - start) * 1000
        if r.status_code == 200:
            d = r.json()
            emotion = d.get("predicted_emotion", "unknown")
            conf = d.get("confidence", 0.0)
            emoji = d.get("emoji", "")
            print(f"{s[:40]:<42} -> {emotion} {emoji:<6} ({conf*100:.1f}%)    {elapsed_ms:.1f}ms")
        else:
            print(f"{s[:40]:<42} -> [HTTP {r.status_code}]: {r.text[:30]}")
    except Exception as err:
        print(f"{s[:40]:<42} -> [ERROR]: {err}")

print("-" * 75)
print("Live test completed successfully.")

