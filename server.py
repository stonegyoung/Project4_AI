from fastapi import FastAPI, Form, BackgroundTasks
import uvicorn
from pydantic import BaseModel

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
import random

import pyrebase
import json
import numpy as np
import re
import pandas as pd

import logging

from dotenv import load_dotenv
load_dotenv()

df = pd.read_csv("quiz2.csv")

theme_list = ["해양쓰레기 발생원인", "해양쓰레기 현황", "해양쓰레기 피해 및 위험성", "해양쓰레기 피해 사례", "태평양 해양 쓰레기 섬", "미세 플라스틱", "허베이스피릿호 원유유출 사고", "검은 공 사건", "약품 사고", "제주 바다 돌고래", "바다 거북", "상괭이"]
datas = [pd.DataFrame() for i in range(len(theme_list))]
for i in range(len(theme_list)):
    if len(df[df['t'] == theme_list[i]]) == 0:
        print(theme_list[i])
    else:
        datas[i] = df[df['t'] == theme_list[i]]


logger1 = logging.getLogger('general_logger')
logger1.setLevel(logging.INFO)  # 최소 레벨을 INFO로 설정
handler1 = logging.FileHandler('info.log')  # history.log에 기록
formatter1 = logging.Formatter('%(asctime)s - %(message)s')
handler1.setFormatter(formatter1)
logger1.addHandler(handler1)

logger2 = logging.getLogger('history_logger')
logger2.setLevel(logging.INFO)  # 최소 레벨을 ERROR로 설정
handler2 = logging.FileHandler('history.log', encoding='utf-8')  # error.log에 기록
formatter2 = logging.Formatter('%(message)s')
handler2.setFormatter(formatter2)
logger2.addHandler(handler2)

with open("./chat/auth.json") as f:
    config = json.load(f)
    
firebase = pyrebase.initialize_app(config)
db = firebase.database()

loginid = set(db.child("User").get().val().keys())
# page_content만 저장
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# json 형식으로 바꾸기
def convert_json(st):
    try:
        st = re.search(r'\{.*?\}', st, re.DOTALL).group(0)
        st = json.loads(st)
        return st
    except:
        return False
    
# 퀴즈 리턴
def return_quiz(n):
    global datas
    sample = datas[n].sample(1).reset_index(drop = True)
    return sample.to_dict(orient = 'records')[0]

# 모델
chatgpt = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature = 1
)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='C:/project4/chat/OceanDB')
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# 챗 메세지
chat_messages = [
    SystemMessage(content='당신은 해양 관련 지식을 가지고 있는 사람입니다. 질문에 대해 100자 내외로 말해주고, 해양이나 해양 생물과 관련된 이야기가 아니면 정중하게 거절해주세요.'),
    HumanMessagePromptTemplate.from_template('{history}'),
    HumanMessagePromptTemplate.from_template('Context: {context}\nQuestion: {ques}')
]

# 챗 프롬프트
chat_prompt = ChatPromptTemplate.from_messages(chat_messages)

##################################################################################################################################################################

app = FastAPI()

class IdPw(BaseModel):
    id : str
    pw : str
    
class Chat(BaseModel):
    id : str
    question : str
    
class QuizId(BaseModel):
    id : str
    
@app.get("/") 
def root(): 
    logger1.info("/")
    return {'result': '접속 완료'}

@app.get("/join") 
def join(member:IdPw):
    # 저장
    global loginid
    if member.id in loginid or member.id+'\u200b' in loginid:
        return {"result": False} # 존재하는 아이디
    data = {"pw" : member.pw, "point": 0, "history" : "", "theme": 11}
    db.child("User").child(member.id).set(data)
    
    loginid = set(db.child("User").get().val().keys())
    
    logger1.info("/join")
    return {'result': True} # 회원가입 완료

@app.get("/login")
def login(member:IdPw):
    # 아이디가 DB안에 있고 pw가 동일하면 로그인 완료
    # pw 다르면 실패
    logger1.info("/login")
    global loginid
    if member.id in loginid: # id 존재
        data = db.child("User").child(member.id).get().val()
        n = np.random.randint(0,len(theme_list))
        if member.pw == data['pw']:
            print(f"[세션ID]: {member.id}")
            db.child("User").child(member.id).update({'theme':n}) # theme 업데이트
            return {'result': theme_list[n]}
        else:
            return {'result': '비밀번호 오류'}
    else:
        return {"result": "아이디 오류"}

@app.get("/chatbot")
async def chatbot(chat:Chat):
    logger1.info("/chatbot")
    if chat.id == '':
        return {"result":False}
    
    data = db.child("User").child(chat.id).get().val()
    theme = theme_list[data['theme']]
    if chat.question == '오늘의 예상 퀴즈':
        result = f"오늘의 퀴즈는 '{theme}' 부분에서 나올 것으로 예상됩니다!\n"
        ans = await chatgpt.ainvoke(f'{theme}에 대해 100자 이내로 알려줘')
        result += ans.content
        return {"result": result}
    else:
        history = data['history']
        context = await retriever.ainvoke(history+chat.question)
        context = format_docs(context)
        
        # 프롬프트 만들기
        result = chat_prompt.invoke({
            'history': history,
            'context': context,
            'ques': chat.question
        })
        
        # 답변
        ans = await chatgpt.ainvoke(result)
        ans = ans.content
        
        # 각 id의 히스토리에 추가
        data = {"history" : history+f'Human: {chat.question}\nAI: {ans}\n'}
        db.child("User").child(chat.id).update(data)
        logger2.info(f"Human: {chat.question}\nAI: {ans}\n\n")
        return {"result": ans}

@app.get("/get_history")
def get_history(id:str=Form(...)):
    logger1.info("/get_history")
    global loginid
    if id in loginid: 
        return {"result": db.child("User").child(id).get().val()['history']}
    else:
        return {"result": False}
    
@app.get("/reset_chat")
def reset_chat(id:str=Form(...)):
    logger1.info("/reset_chat")
    # 초기화 코드
    global loginid
    if id in loginid: 
        db.child("User").child(id).update({"history": ""})
        return {"result": True} # 초기화 완료
    else:
        return {"result": False}
    
@app.get("/qna")
async def qna(qid:QuizId):
    logger1.info("/qna")
    n = db.child("User").child(qid.id).get().val()['theme']
    json_quiz = return_quiz(n)
    return json_quiz

@app.get("/testchatbot")
def testchatbot(tc:Chat):
    logger1.info("/testchatbot")
    n = np.random.randint(2, 7)
    st = '테스트용 챗봇입니다.\n'
    
    return {"result":st*n}

@app.get("/testqna")
def testqna(qid:QuizId):
    logger1.info("/testqna")
    js ={
        "q": "드라마의 특성으로 거리가 먼 것은?",
        "n1" :"장소의 제한을 거의 받지 않는다.",
        "n2" :"음악을 통하여 분위기를 알 수 있다.",
        "n3" :"주로 문자를 통하여 내용이 전달된다.",
        "n4" :"연출가, 작가, 배우 등 여러 사람이 함께 만든다.",
        "a" :"3",
        "t" : "드라마",
        "s" : "드라마는 문자보다는 음성과 영상으로 전달됩니다."
    }
    return js
    
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9100)