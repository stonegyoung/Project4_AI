from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough

import numpy as np

import re
import json

from dotenv import load_dotenv
load_dotenv()

def convert_json(st):
    try:
        st = re.search(r'\{.*?\}', st, re.DOTALL).group(0)
        st = json.loads(st)
        return st
    except:
        return False
    
#########################################################################################################################
# 질문 만들기

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
    "'문제' : '{question}', 'n1': '{n1}', 'n2': '{n2}', 'n3': '{n3}', 'n4': '{n4}', '정답': '{answer}'" # examples랑 같아야 함
)
# print(example_prompt)
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix = "당신은 선생님입니다. {context}를 기반으로 정확하게 {theme}에 대한 문제를 하나 만들고 4가지 선다와 정답을 JSON 형식으로 만들어주세요. 정답이 없을 수는 없고, 정답이 아닌 선지는 답과 거리가 멀어야 합니다",
    suffix="'ques': '생성된 문제', 'n1': '선택지 1', 'n2': '선택지 2', 'n3': '선택지 3', 'n4': '선택지 4', 'ans': '정답 번호'",
    input_variables=["theme"],
)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='C:/project4/chat/testDB')
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# page_content만 저장
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chatgpt = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature = 1
)

rag_chain = (
    {'context': retriever|format_docs, 'theme': RunnablePassthrough()} # chat_prompt가 갖는 dict
    |prompt # 프롬프트
    |chatgpt # 모델
)

theme_list = ["해양쓰레기 종류", "해양쓰레기 발생원인", "해양쓰레기 현황", "해양쓰레기 피해 및 위험성"]
n = np.random.randint(0,len(theme_list))
ans = rag_chain.invoke(theme_list[n])
js = convert_json(ans.content)
print("before")
print(js)

#######################################################################################################################################################################
# 검증 하기
val_examples = [
    {
        "question":"해양쓰레기 종류에 대한 설명으로 알맞은 것은?",
        "n1": "해양쓰레기는 육상과 해상 모두에서 발생할 수 있다.",
        "n2": "해양쓰레기는 오직 플라스틱으로만 구성된다.",
        "n3": "해양쓰레기는 바다 생물의 생태계에 직접적인 영향을 미치지 않는다.",
        "n4": "해양쓰레기는 주로 인간의 기호에 따라 생성된다.",
        "answer": "1",
        "corr_n1": "해양쓰레기는 오직 플라스틱으로만 구성된다.",
        "corr_n2": "미세 플라스틱이 해양 생물의 성장과 번식에 영향을 준다.",
        "corr_n3": "해양쓰레기는 바다 생물의 생태계에 직접적인 영향을 미치지 않는다.",
        "corr_n4": "해양쓰레기는 주로 인간의 기호에 따라 생성된다.",
    },
    {
        "question": "해양쓰레기가 발생하는 주된 원인은 무엇인가?",
        "n1": "하천과 강을 따라 바다로 들어오는 쓰레기",
        "n2": "해양 동물의 생태계를 보호하기 위한 법제정",
        "n3": "태풍이나 폭우로 인해 쓰레기가 바다로 이동",
        "n4": "바다에서 태어나는 새로운 종류의 쓰레기",
        "answer": "1",
        "corr_n1": "하천과 강을 따라 바다로 들어오는 쓰레기", 
        "corr_n2": "해양 동물의 생태계를 보호하기 위한 법제정", 
        "corr_n3": "멸종 위기 종의 증가",
        "corr_n4": "자원봉사자의 봉사활동",
    }
]
val_example_prompt = PromptTemplate.from_template(
    "Before valid\n"
    "'문제' : '{question}', 'n1': '{n1}', 'n2': '{n2}', 'n3': '{n3}', 'n4': '{n4}', '정답': '{answer}'"
    "\nAfter valid\n"
    "'문제' : '{question}', 'n1': '{corr_n1}', 'n2': '{corr_n2}', 'n3': '{corr_n3}', 'n4': '{corr_n4}', 'answer': '{answer}'"
)

sf = '''Before valid
'문제' : '{ques}', 'n1': '{n1}', 'n2': '{n2}', 'n3': '{n3}', 'n4': '{n4}', '정답': '{ans}'
After valid
'''

valid_prompt = FewShotPromptTemplate(
    examples=val_examples,
    example_prompt=val_example_prompt,
    prefix="당신은 다음과 같은 퀴즈를 보고, 문제에 대한 오답 선지가 명확히 오답인지 검증하는 사람입니다. 정답을 제외하고, 오해의 소지가 있는 오답은 명확한 오답으로 변경해서 JSON 형식으로 만들어주세요.",
    suffix= sf,
    input_variables=["ques", "n1", "n2", "n3", "n4", "ans"],
)

val_chain = (
    RunnableMap({
        'ques': RunnablePassthrough(),
        'n1': RunnablePassthrough(),
        'n2': RunnablePassthrough(),
        'n3': RunnablePassthrough(),
        'n4': RunnablePassthrough(),
        'ans': RunnablePassthrough(),
    })
    |valid_prompt | chatgpt 
)
output = val_chain.invoke(js)
result = convert_json(output.content)
print("after")
print(result)