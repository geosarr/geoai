import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Header
import ollama

load_dotenv()

app = FastAPI()

API_KEY_CREDITS = {os.getenv("API_KEY"): 5}


def verify_key(api_key: str = Header(None)):
    if API_KEY_CREDITS.get(api_key, 0) <= 0:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


@app.post("/predict")
def predict(prompt: str, api_key: str = Depends(verify_key)):
    API_KEY_CREDITS[api_key] -= 1
    # Simulate a call to the ollama API
    # In a real-world scenario, you would replace this with the actual API call
    # For example:
    # 1. Call the ollama API with the prompt
    # 2. Get the response
    # 3. Return the response to the client
    # For now, we'll just return a dummy response
    # Dummy response
    response = {"message": {"content": prompt}}
    # ollama.chat(
    #     model="mistral",
    #     messages=[{"role": "user", "content": prompt}],
    # )
    return {"response": response["message"]["content"]}
