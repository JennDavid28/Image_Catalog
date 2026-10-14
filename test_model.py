
import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

response = client.models.generate_content(
    model="gemini-3.1-flash-image",
    contents="Generate a realistic red apple on a wooden table.",
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"]
    )
)

for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("test_generated.png")
        print("Image generated successfully: test_generated.png")
        break
else:
    print("No image was returned.")
