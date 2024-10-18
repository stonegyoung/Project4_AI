from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/login")
def login():
    return {"result": "로그인 기능입니다"}

@app.get("/chatbot")
def chatbot():
    return {"result": "챗봇 기능입니다"}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9100)
    