"""
check_models.py
----------------
Lists every model your GROQ_API_KEY currently has access to.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY not found in your .env file.")
else:
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    if response.status_code != 200:
        print(f"Request failed ({response.status_code}): {response.text}")
    else:
        models = response.json().get("data", [])
        print(f"You have access to {len(models)} model(s):\n")
        for m in models:
            print(f"  - {m['id']}")