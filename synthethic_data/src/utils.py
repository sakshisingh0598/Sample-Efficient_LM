import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEYS.clear()
i = 1
while True:
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key:
        GEMINI_API_KEYS.append(key)
        i += 1
    else:
        single_key = os.getenv("GEMINI_API_KEY")
        if single_key and not GEMINI_API_KEYS:
            GEMINI_API_KEYS.append(single_key)
        break

if not GEMINI_API_KEYS:
    raise ValueError("No Gemini API Key found in .env file. Please set GEMINI_API_KEY or GEMINI_API_KEY_1, etc.")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
