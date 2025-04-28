from fastapi import FastAPI

app = FastAPI()


@app.get("/predict")
def predict(prompt: str):
    return {f"message": "This is a prediction {prompt} \n."}


print(predict("test"))
