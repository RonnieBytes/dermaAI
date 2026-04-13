from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv
import os
import base64

load_dotenv()

app = FastAPI(title="DermAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_skin_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        mime_type = f"image/{ext}" if ext != 'jpg' else 'image/jpeg'

        prompt = """You are an expert dermatologist AI.
Analyze the uploaded skin image and respond in this exact structured format:

**Diagnosis:** Healthy / Normal Skin or [Specific Condition Name]

**Confidence:** High / Medium / Low

**Analysis:**
- Brief description of what is seen
- Likely causes or contributing factors
- Recommended treatments or medicines
- Home care / Skincare tips
- Red flags - When to see a dermatologist immediately

Always end with: "This is an AI-generated analysis for educational purposes only. Please consult a qualified dermatologist for proper medical advice.""""

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }],
            max_tokens=1024,
            temperature=0.3
        )

        return {
            "full_report": response.choices[0].message.content,
            "note": "Powered by Groq Llama-4 Scout Vision • Not a medical diagnosis"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
