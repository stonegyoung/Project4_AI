from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ''
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

store = {}
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


while True:
    id = 'rag123'
    history = get_session_history(id)
    try:
        q = input()
        
        context = format_docs(retriever.invoke(history+q))
        result = chat_prompt.invoke({
            'history':history,
            'context': context,
            'ques': q
        })

        ans = chatgpt.invoke(result).content
        store[id] += f'Human: {q}\nAI: {ans}\n'

        print(ans)
    except:
        print(store[id])
        break