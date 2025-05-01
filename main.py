from dotenv import find_dotenv, load_dotenv
from dotenv.main import DotEnv
from fastapi import FastAPI, Depends, HTTPException, Header
from langchain_openai import OpenAI
from constants import API_KEYS
from rag import create_prompt


if not len(API_KEYS):
    raise ValueError("No API keys found")

API_KEY_CREDITS = dict.fromkeys(API_KEYS.values(), 5)

load_dotenv()
app = FastAPI()
OPEN_AI_LLM = OpenAI()


def get_response(request: str, llm: OpenAI):
    if isinstance(llm, OpenAI):
        return llm.invoke(request)
    return request


def verify_key(x_api_key: str = Header(None)):
    if API_KEY_CREDITS.get(x_api_key, 0) <= 0:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


@app.post("/predict")
def predict(request: str, x_api_key: str = Depends(verify_key)):
    API_KEY_CREDITS[x_api_key] -= 1
    prompt = create_prompt(request)
    return {"response": get_response(prompt, OPEN_AI_LLM)}


if __name__ == "__main__":
    print(get_response("What is rust langage ?", OpenAI()))
