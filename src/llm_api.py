import json
import time
import urllib.error
import urllib.request


def _normalize_ollama_host(host: str) -> str:
    return host.rstrip('/')


def ollama_completion(prompt, model="llama3.1:8b", host="http://127.0.0.1:11434",
                      max_tokens=700, temperature=0.0, timeout=180, retries=5):
    endpoint = f"{_normalize_ollama_host(host)}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                return response_data.get("response", "")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(f"Failed to query Ollama model '{model}' at {endpoint}: {last_error}")
