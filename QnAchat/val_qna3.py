from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
import random
from langchain_core.prompts.few_shot import FewShotPromptTemplate
import re
import json

import numpy as np
def convert_json(st):
    try:
        st = re.search(r'\{.*?\}', st, re.DOTALL).group(0)
        st = json.loads(st)
        return st
    except:
        return False
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chatgpt = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

class Quiz(BaseModel):
    ques: str = Field(description="객관식으로 선택할 수 있는 문제")
    ans: str = Field(description="문제에 대한 정답")
    
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='C:/project4/chat/testDB')
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 선생님입니다. {context}를 기반으로 정확하게 {theme}에 대한 객관식으로 선택할 수 있는 문제와 문제에 대한 정답을 만들어주세요. 문제는 이의 제기가 일어나지 않도록 확실한 정답이 아니라면 '거리가 가까운 것은?', '거리가 먼 것은?'이라는 형태로 끝나야 합니다."),
        ("user", "#Format: {format_instructions}"),
    ]
)

parser = JsonOutputParser(pydantic_object=Quiz)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

chain = (
    {'context': retriever|format_docs, 'theme': RunnablePassthrough()} # chat_prompt가 갖는 dict
    |prompt | chatgpt | parser)  

theme_list = ["해양쓰레기 종류", "해양쓰레기 발생원인", "해양쓰레기 현황", "해양쓰레기 피해 및 위험성"]
n = np.random.randint(0,len(theme_list))
js_ans = chain.invoke(theme_list[n])
print(js_ans)
############################################################################################33
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
    prefix="당신은 다음의 문제를 보고, 문제에 대한 오답 선지를 만드는 사람입니다. 오해의 소지가 있을 문구는 제외하여 오답을 만들고 JSON 형식으로 만들어주세요.",
    suffix= "문제: {ques}\n정답: {ans}\n오답: ",
    input_variables=["ques", "ans"],
)

val_chain = (
    RunnableMap({
        'ques': RunnablePassthrough(),
        'ans': RunnablePassthrough(),
    })
    |valid_prompt | chatgpt
)

wrong = val_chain.invoke(js_ans)
result = convert_json(wrong.content)

options = [js_ans['ans']] + list(result.values())
random.shuffle(options)

quiz = {"문제": js_ans['ques'], "n1":options[0], "n2":options[1], "n3": options[2], "n4": options[3], "정답": options.index(js_ans['ans'])+1}

print(quiz)
