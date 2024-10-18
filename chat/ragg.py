from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

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
    SystemMessage(content='당신은 해양 관련 지식을 가지고 있는 사람입니다. 질문에 대해 100자 내외로 말해주고, 해양과 관련된 이야기가 아니면 대답하지 말아주세요.'),
    HumanMessagePromptTemplate.from_template('Context: {context}\nQuestion: {ques}')
]

# 챗 프롬프트
chat_prompt = ChatPromptTemplate.from_messages(chat_messages)

# retriever|format_docs -> 질문을 검색기에 전달 후 document 객체 생성 -> page_content만 저장
# RunnablePassthrough -> 질문을 그대로 넣는다
# 임베딩 벡터를 디비에 저장
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='./chat/testDB')
retriever = vectorstore.as_retriever(search_kwargs={"k":3})
rag_chain = (
    {'context': retriever|format_docs, 'ques': RunnablePassthrough()} # chat_prompt가 갖는 dict
    |chat_prompt # 프롬프트
    |chatgpt # 모델
)

q = input()
result = rag_chain.invoke(q)
print(result.content)