"""Fetch all ElevenLabs shared voice metadata (no audio). Stores to el_voices.json."""
import json
import os
import time
import requests

API_KEY = os.environ.get("EL_API_KEY", "")
if not API_KEY:
    raise SystemExit("ERROR: EL_API_KEY environment variable not set")
HEADERS = {"xi-api-key": API_KEY}
OUT = "/root/fish-speech/el_voices.json"

all_voices = []
page = 0
last_sort_id = None

print("Fetching ElevenLabs voice metadata...")
while True:
    params = {"page_size": 100, "page": page}
    if last_sort_id:
        params["last_sort_id"] = last_sort_id

    resp = requests.get(
        "https://api.elevenlabs.io/v1/shared-voices",
        headers=HEADERS, params=params, timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    voices = data.get("voices", [])
    if not voices:
        break

    all_voices.extend(voices)
    print(f"  Page {page}: {len(voices)} voices (total: {len(all_voices)})", flush=True)

    if not data.get("has_more"):
        break

    last_sort_id = data.get("last_sort_id")
    page += 1
    time.sleep(0.2)

# Keep only what we need
slim = []
for v in all_voices:
    slim.append({
        "id":          v.get("voice_id", ""),
        "name":        v.get("name", ""),
        "description": (v.get("description") or "").strip(),
        "preview_url": v.get("preview_url", ""),
        "gender":      v.get("gender", ""),
        "age":         v.get("age", ""),
        "accent":      v.get("accent", ""),
        "language":    v.get("language", ""),
        "category":    v.get("category", ""),
    })

with open(OUT, "w") as f:
    json.dump(slim, f)

print(f"\nSaved {len(slim)} voices to {OUT}")
