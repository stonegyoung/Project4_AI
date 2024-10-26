from fastapi import FastAPI, Form
import uvicorn
from pydantic import BaseModel


from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import pyrebase
import json

from dotenv import load_dotenv

load_dotenv()


with open("./chat/auth.json") as f:
    config = json.load(f)
    
firebase = pyrebase.initialize_app(config)
db = firebase.database()

def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if db.child("User").child(session_ids).get().val() is None:  # 세션 ID가 data에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 data에 저장
        data = {"history" : ""}
        db.child("User").child(session_ids).set(data) 
    return db.child("User").child(session_ids).get().val()['history'] # 해당 세션 ID에 대한 세션 기록 반환

# page_content만 저장
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 모델
chatgpt = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature = 0.3
)

# 챗 메세지
chat_messages = [
    SystemMessage(content='당신은 해양 관련 지식을 가지고 있는 사람입니다. 질문에 대해 100자 내외로 말해주고, 해양과 관련된 이야기가 아니면 정중하게 거절해주세요.'),
    HumanMessagePromptTemplate.from_template('{history}'),
    HumanMessagePromptTemplate.from_template('Context: {context}\nQuestion: {ques}')
]

# 챗 프롬프트
chat_prompt = ChatPromptTemplate.from_messages(chat_messages)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='C:/project4/chat/testDB')
retriever = vectorstore.as_retriever(search_kwargs={"k":3})


app = FastAPI()

class Login(BaseModel):
    id : str
    pw : str
    
class Chat(BaseModel):
    id : str
    question : str


@app.get("/login")
def login(login:Login):
    # 아이디가 DB안에 있고 pw가 동일하면 로그인 완료
    # DB 안에 없으면 회원가입/그냥 바로 만들기
    # pw 다르면 실패
    return {"result": "로그인 기능입니다"}

@app.get("/chatbot")
def chatbot(chat:Chat):
    # 나중에 비동기
    history = get_session_history(chat.id)
    context = format_docs(retriever.invoke(history+chat.question))
    
    # 프롬프트 만들기
    result = chat_prompt.invoke({
        'history': history,
        'context': context,
        'ques': chat.question
    })
    
    # 답변
    ans = chatgpt.invoke(result).content
    # 각 id의 히스토리에 추가
    data = {"history" : history+f'Human: {chat.question}\nAI: {ans}\n'}
    db.child("User").child(chat.id).update(data)
    return {"result": ans}

@app.get("/reset_chat")
def reset_chat(id:str=Form(...)):
    # 초기화 코드
    db.child("User").child(id).update({"history": ""})
    return {"result": f"{id}의 챗봇 내역이 초기화되었습니다."}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9100)
    