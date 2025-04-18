# synthetic-chatgen

Generate high-quality, multi-persona conversational data with Gemini.

## Setup

1. `python3 -m venv venv && source venv/bin/activate`  
2. `pip install -r requirements.txt` (This will install `google-generativeai` and `python-dotenv`)
3. Create a `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   # GEMINI_MODEL=gemini-2.0-flash (Optional, defaults to this in the code)
   ```

## Usage

```bash
python src/generate.py
```

### Output

Generated dialogues will be in `outputs/dialogues.json` as a single JSON array.