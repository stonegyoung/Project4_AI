from fastapi import FastAPI, Form
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

################################################################################################################################################################
# 질문과 정답
examples = [
    {
        "question": "드라마의 특성으로 거리가 먼 것은?",
        "answer" :"주로 문자를 통하여 내용이 전달된다.",
    },
    {
        "question": "소설 *소나기*에서 소년이 소녀에게 특별한 감정을 느끼게 된 계기로 거리가 가까운 것은?",
        "answer": "둘이 함께 소나기를 피하면서"
    }
]
example_prompt = PromptTemplate.from_template(
    "'ques' : '{question}', 'ans': '{answer}'" # examples랑 같아야 함
)
qna_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix = "당신은 선생님입니다. {context}를 기반으로 정확하게 {theme}에 대한 객관식으로 선택할 수 있는 문제를 하나 만들고 질문에 대한 정답을 JSON 형식으로 만들어주세요. 문제는 이의 제기가 일어나지 않게 '거리가 가까운 것은?', '거리가 먼 것은?'이라는 형태로 끝나야 합니다.",
    suffix="'ques': '생성된 문제', 'ans': '정답'",
    input_variables=["theme"],
)
rag_chain = (
    {'context': retriever|format_docs, 'theme': RunnablePassthrough()} # chat_prompt가 갖는 dict
    |qna_prompt # 프롬프트
    |chatgpt # 모델
)

theme_list = ["해양쓰레기 발생원인", "해양쓰레기 현황", "해양쓰레기 피해 및 위험성", "해양쓰레기 피해 사례", "태평양 해양 쓰레기 섬", "미세 플라스틱", "허베이스피릿호 원유유출 사고", "검은 공 사건", "약품 사고", "제주 바다 돌고래", "바다 거북", "상괭이"]


# 오답
val_examples = [
    {
        "question":"해양쓰레기 종류에 대한 설명으로 알맞은 것은?",
        "answer": "해양쓰레기는 육상과 해상 모두에서 발생할 수 있다.",
        "corr_1": "해양쓰레기는 오직 플라스틱으로만 구성된다.",
        "corr_2": "해양쓰레기는 바다 생물의 생태계에 직접적인 영향을 미치지 않는다.",
        "corr_3": "해양쓰레기는 모두 수거되어 재활용된다.",
    },
    {
        "question": "해양쓰레기가 발생하는 주된 원인은 무엇인가?",
        "answer": "하천과 강을 따라 바다로 들어오는 쓰레기", 
        "corr_1": "해양 동물의 생태계를 보호하기 위한 법제정", 
        "corr_2": "멸종 위기 종의 증가",
        "corr_3": "자원봉사자의 봉사활동",
    }
]
val_example_prompt = PromptTemplate.from_template(
    "문제: {question}\n정답: {answer}\n"
    "오답: 'wrong1': '{corr_1}', 'wrong2': '{corr_2}', 'wrong3': '{corr_3}'"
)

valid_prompt = FewShotPromptTemplate(
    examples=val_examples,
    example_prompt=val_example_prompt,
    prefix="당신은 다음의 문제를 보고, 문제에 대한 오답 선지를 만드는 사람입니다. 오답은 정답과 비슷한 형식으로 만들되, 오해의 소지가 있을 문구는 제외하여 만들어주세요. 결과는 JSON 형식으로 만들어주세요.",
    suffix= "문제: {ques}\n정답: {ans}\n오답: ",
    input_variables=["ques", "ans"],
)

wrong_chain = (
    RunnableMap({
        'ques': RunnablePassthrough(),
        'ans': RunnablePassthrough(),
    })
    |valid_prompt | chatgpt 
)
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
async def qna():
    logger1.info("/qna")
    global theme_list
    n = np.random.randint(0,len(theme_list))
    ans = await rag_chain.ainvoke(theme_list[n])
    js = convert_json(ans.content)
    
    ans = await wrong_chain.ainvoke(js)
    wrong = convert_json(ans.content)
    
    options = [js['ans']] + list(wrong.values())
    random.shuffle(options)

    quiz = {"문제": js['ques'], "n1":options[0], "n2":options[1], "n3": options[2], "n4": options[3], "정답": options.index(js['ans'])+1}

    return quiz

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
    uvicorn.run(app, host='0.0.0.0', port=9200)