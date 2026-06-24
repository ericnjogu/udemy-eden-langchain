from pathlib import Path
from ollama import chat
from dotenv import load_dotenv
import os

load_dotenv()

for img in sorted(Path(os.environ["PDF_DIR"]).glob("maxys-*.png")):
    response = chat(
        model="gemma4:12b",
        messages=[{
            "role": "user",
            "content": "Extract all text exactly as written.",
            "images": [str(img)]
        }]
    )

    print(f"=== {img} ===")
    print(response)
    print(response["message"]["content"])