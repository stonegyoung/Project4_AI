from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
import random

import json
import numpy as np
import re
import csv


from dotenv import load_dotenv
load_dotenv()

output_file = "quiz1.csv"
file = open(output_file, "w", encoding="utf-8", newline="")
# 컬럼 이름 설정
columns = ["q", "n1", "n2", "n3", "n4", "a", "t", "s"]
writer = csv.DictWriter(file, fieldnames=columns)
writer.writeheader()  # 헤더 작성

theme_list = ["해양쓰레기 발생원인", "해양쓰레기 현황", "해양쓰레기 피해 및 위험성", "해양쓰레기 피해 사례", "태평양 쓰레기섬", "미세플라스틱", "허베이스피릿호 원유유출 사고", "호주 검은 공 사건", "약품 사고", "폐어구에 걸린 돌고래", "우리나라 바다 거북", "상괭이"]

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
    
# 퀴즈 생성
def make_quiz(n):
    ans = rag_chain.invoke(theme_list[n])
    js = convert_json(ans.content)
    
    try:
        ans = wrong_chain.invoke(js)
        wrong = convert_json(ans.content)
        
        q = f"'{js['ques']}' 문제에 대해 '{js['ans']}'가 정답인 이유가 뭐야? 30자 이내로 격식체로 설명해줘"
        # print(q)
        ans = chatgpt.invoke(q)
        explain = ans.content
    
        options = [js['ans']] + list(wrong.values())
        random.shuffle(options)

        right = options.index(js['ans'])+1
        data = {
            "q": js['ques'],
            "n1": options[0],
            "n2": options[1],
            "n3": options[2],
            "n4": options[3],
            "a": right,
            "t": theme_list[n],
            "s": explain
        }
        print(data)
        writer.writerow(data)
        # quiz = {"q": js['ques'], "n1":options[0], "n2":options[1], "n3": options[2], "n4": options[3], "a": right, "t": theme_list[n], "s": js['explain']}
        # return quiz
    except:
        pass

# 모델
chatgpt = ChatOpenAI(
    model_name="gpt-4o",
    temperature = 1
)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='C:/project4/chat/OceanDB')
retriever = vectorstore.as_retriever(search_kwargs={"k":5})


################################################################################################################################################################
# 질문과 정답
examples = [
    {
        "question": "드라마의 특성으로 알맞은 것은?",
        "answer" :"주로 영상을 통하여 내용이 전달된다.",
    },
    {
        "question": "소설 *소나기*에서 소년이 소녀에게 특별한 감정을 느끼게 된 계기로 옳은 것은?",
        "answer": "둘이 함께 소나기를 피하면서"
    }
]
example_prompt = PromptTemplate.from_template(
    "'ques' : '{question}', 'ans': '{answer}'" # examples랑 같아야 함
)
qna_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt, 
    prefix = "당신은 선생님입니다. {context}를 기반으로 정확하게 {theme}에 대한 객관식으로 선택할 수 있는 문제를 하나 만들고 질문에 대한 정답을 JSON 형식으로 만들어주세요. '옳은 것은?' 혹은 '알맞은 것은?' 으로 끝나는 문제를 만들고, 정답은 이의 제기가 일어나지 않게 만들어주세요.",
    suffix="'ques': '생성된 문제', 'ans': '정답'",
    input_variables=["theme"],
)
rag_chain = (
    {'context': retriever|format_docs, 'theme': RunnablePassthrough()} # chat_prompt가 갖는 dict
    |qna_prompt # 프롬프트
    |chatgpt # 모델
)

explain_chain = (
    retriever|format_docs # chat_prompt가 갖는 dict
    |chatgpt # 모델
)

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
        "question": "다음 중 물의 성질에 대한 설명으로 옳은 것은?",
        "answer": "물은 대부분의 물질을 녹이는 우수한 용매이다.", 
        "corr_1": "물은 검은 색이다.", 
        "corr_2": "물은 100℃에서 얼고, 0℃에서 끓는다.",
        "corr_3": "물은 극성 물질과 이온성 물질을 녹이지 못한다.",
    },
    {
        "question": "대한민국의 수도로 옳은 것은?",
        "answer": "서울", 
        "corr_1": "부산", 
        "corr_2": "인천",
        "corr_3": "대전",
    }
]

val_example_prompt = PromptTemplate.from_template(
    "문제: {question}\n정답: {answer}\n"
    "오답: 'wrong1': '{corr_1}', 'wrong2': '{corr_2}', 'wrong3': '{corr_3}'"
)

valid_prompt = FewShotPromptTemplate(
    examples=val_examples,
    example_prompt=val_example_prompt,
    prefix="""다음은 문제와 정답입니다. 이 정보를 기반으로, 3개의 오답을 생성하세요.
    가능한 다양하게 구성되도록 만들고, 문제의 맥락에서 명백히 잘못된 내용을 포함해야 합니다.
    결과는 JSON 형식으로 만들어주세요.\n""",
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

for n in range(len(theme_list)):
    for _ in range(1):
        make_quiz(n)
        

file.close()