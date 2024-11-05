from fastapi import FastAPI, Form
import uvicorn
from pydantic import BaseModel

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate

import pyrebase
import json
import numpy as np
import re

import logging

from dotenv import load_dotenv
load_dotenv()

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

def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if db.child("User").child(session_ids).get().val() is None:  # 세션 ID가 data에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 data에 저장
        data = {"pw":"", "point":0, "history" : ""}
        db.child("User").child(session_ids).set(data) 
    return db.child("User").child(session_ids).get().val()['history'] # 해당 세션 ID에 대한 세션 기록 반환

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

# 모델
chatgpt = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature = 0.3
)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='C:/project4/chat/testDB')
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# 챗 메세지
chat_messages = [
    SystemMessage(content='당신은 해양 관련 지식을 가지고 있는 사람입니다. 질문에 대해 100자 내외로 말해주고, 해양과 관련된 이야기가 아니면 정중하게 거절해주세요.'),
    HumanMessagePromptTemplate.from_template('{history}'),
    HumanMessagePromptTemplate.from_template('Context: {context}\nQuestion: {ques}')
]

# 챗 프롬프트
chat_prompt = ChatPromptTemplate.from_messages(chat_messages)

################################################################################################################################################################
# 질문 예시
examples = [
    {
        "question": "드라마의 특성으로 거리가 먼 것은?",
        "n1" :"장소의 제한을 거의 받지 않는다.",
        "n2" :"음악을 통하여 분위기를 알 수 있다.",
        "n3" :"주로 문자를 통하여 내용이 전달된다.",
        "n4" :"연출가, 작가, 배우 등 여러 사람이 함께 만든다.",
        "answer" :"3"
    },
    {
        "question": "독도가 대한민국 영토임을 알리는 이유로 알맞은 것은",
        "n1": "독도가 일본 본토와 가깝기 때문이다.",
        "n2": "독도는 중요한 자원이 많아 경제적으로 중요한 지역이다.",
        "n3": "독도가 오랫동안 대한민국의 행정 구역으로 관리되어 왔기 때문이다.",
        "n4": "독도에는 대한민국의 유명한 관광지가 있기 때문이다.",
        "answer": "3"
    },
    {
        "question": "소설 *소나기*에서 소년이 소녀에게 특별한 감정을 느끼게 된 계기로 거리가 가까운 것은?",
        "n1": "둘이 함께 소나기를 피하면서",
        "n2": "소녀가 소년에게 꽃다발을 주면서",
        "n3": "소년이 소녀의 생일에 선물을 주면서",
        "n4": "소녀가 학교에서 발표를 잘해서",
        "answer": "1"
    }
]
example_prompt = PromptTemplate.from_template(
    "'문제' : '{question}', 'n1': '{n1}', 'n2': '{n2}', 'n3': '{n3}', 'n4': '{n4}', '정답': '{answer}'"
)
qna_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix = "당신은 선생님입니다. {context}를 기반으로 정확하게 {theme}에 대한 문제를 하나 만들고 4가지 선다와 정답을 JSON 형식으로 만들어주세요. 정답이 없을 수는 없고, 정답이 아닌 선지는 답과 거리가 멀어야 합니다",
    suffix="'문제': '생성된 문제', 'n1': '선택지 1', 'n2': '선택지 2', 'n3': '선택지 3', 'n4': '선택지 4', '정답': '정답 번호'",
    input_variables=["theme"],
)

rag_chain = (
    {'context': retriever|format_docs, 'theme': RunnablePassthrough()} # chat_prompt가 갖는 dict
    |qna_prompt # 프롬프트
    |chatgpt # 모델
)

theme_list = ["해양쓰레기 종류", "해양쓰레기 발생원인", "해양쓰레기 현황", "해양쓰레기 피해 및 위험성"]
################################################################################################################################################################


app = FastAPI()

class IdPw(BaseModel):
    id : str
    pw : str
    
class Chat(BaseModel):
    id : str
    question : str
    
@app.get("/") 
def root(): 
    logger1.info("/")
    return {'result': '접속 완료'}

@app.get("/join") 
def join(member:IdPw):
    # 저장
    global loginid
    if member.id in loginid:
        return {"result": False} # 존재하는 아이디
    data = {"pw" : member.pw, "point": 0, "history" : ""}
    db.child("User").child(member.id).set(data)
    
    loginid = set(db.child("User").get().val().keys())
    
    logger1.info("/join")
    return {'result': True} # 회원가입 완료

@app.get("/login")
def login(member:IdPw):
    # 아이디가 DB안에 있고 pw가 동일하면 로그인 완료
    # DB 안에 없으면 회원가입/그냥 바로 만들기
    # pw 다르면 실패
    logger1.info("/login")
    global loginid
    if member.id in loginid: # id 존재
        if member.pw == db.child("User").child(member.id).get().val()['pw']:
            return {'result': True}
        else:
            return {'result': '비밀번호 오류'}
    else:
        return {"result": "아이디 오류"}

@app.get("/chatbot")
async def chatbot(chat:Chat):
    # 나중에 비동기
    logger1.info("/chatbot")
    if chat.id == '':
        return {"result":False}
    history = get_session_history(chat.id)
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
def qna():
    logger1.info("/qna")
    global theme_list
    n = np.random.randint(0,len(theme_list))
    ans = rag_chain.invoke(theme_list[n])
    js = convert_json(ans.content)
    return js

@app.get("/testchatbot")
def testchatbot(tc:Chat):
    logger1.info("/testchatbot")
    n = np.random.randint(2, 7)
    st = '테스트용 챗봇입니다.\n'
    
    return {"result":st*n}

@app.get("/testqna")
def testqna():
    logger1.info("/testqna")
    st ='''{
        "문제": "드라마의 특성으로 거리가 먼 것은?",
        "n1" :"장소의 제한을 거의 받지 않는다.",
        "n2" :"음악을 통하여 분위기를 알 수 있다.",
        "n3" :"주로 문자를 통하여 내용이 전달된다.",
        "n4" :"연출가, 작가, 배우 등 여러 사람이 함께 만든다.",
        "정답" :"3"
    }'''
    js = convert_json(st)
    return js

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9100)