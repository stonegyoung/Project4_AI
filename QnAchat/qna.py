from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
# from langchain.prompts import PromptTemplate

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
    
    
examples = [
    {
        "question": "염종이 첨성대를 부순 이유로 알맞은것은?",
        "n1" :"첨성대가 실제와 달라서",
        "n2" :"첨성대가 자꾸 기울어져서",
        "n3" :"아랑이 자기 지시대로 만들지 않아서",
        "n4" :"아랑이 첨성대를 자기가 만든 것이라고 선덕여양에게 말해 달라고 해서",
        "answer" :"4"
    },
    {
        "question": "드라마의 특성으로 알맞지 않은 것은?",
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
        "question": "소설 *소나기*에서 소년이 소녀에게 특별한 감정을 느끼게 된 계기로 알맞은 것은?",
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
# print(example_prompt)
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix = "당신은 선생님입니다. {context}를 기반으로 정확하게 {theme}에 대한 문제를 하나 만들고 4가지 선다와 정답을 JSON 형식으로 만들어주세요. 정답이 없을 수는 없고, 정답이 아닌 선지에 대해서는 이의가 없어야 합니다",
    suffix="'문제': '생성된 문제', 'n1': '선택지 1', 'n2': '선택지 2', 'n3': '선택지 3', 'n4': '선택지 4', '정답': '정답 번호'",
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

ans = rag_chain.invoke("해양쓰레기 종류")
js = convert_json(ans.content)
print(js)