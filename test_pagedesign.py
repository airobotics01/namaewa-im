import streamlit as st
import openai
import json
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# 환경 변수 로드 (OPENAI API 키)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ JSON 파일에 대화 내역 저장
CHAT_HISTORY_FILE = "chat_history.json"

def save_memory_to_json(memory, filename=CHAT_HISTORY_FILE):
    """ConversationBufferMemory의 데이터를 JSON 파일로 저장"""
    data = memory.load_memory_variables({})
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_memory_from_json(memory, filename=CHAT_HISTORY_FILE):
    """JSON 파일에서 대화 내역을 불러와 ConversationBufferMemory에 저장"""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "history" in data:
            messages = data["history"].split("\n")
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    memory.save_context(
                        {"input": messages[i].replace("Human: ", "").strip()},
                        {"output": messages[i+1].replace("AI: ", "").strip()},
                    )

# ✅ Streamlit UI 설정
st.set_page_config(page_title="LangChain Chatbot", layout="wide")
st.title("🤖 ChatGPT 스타일 챗봇")

# ✅ 대화 기록 저장을 위한 LangChain Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
    load_memory_from_json(st.session_state.memory)  # JSON에서 불러오기

memory = st.session_state.memory

# ✅ LLM 모델 설정 (GPT-4o 사용)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# ✅ Streamlit 채팅 인터페이스
st.write("💬 **챗봇과 대화하세요!**")

# ✅ 기존 대화 기록 불러오기
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 받기
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # GPT 응답 생성
    with st.chat_message("assistant"):
        response = llm.predict(user_input)
        st.markdown(response)

    # LangChain Memory에 저장
    memory.save_context({"input": user_input}, {"output": response})
    
    # JSON 파일에도 저장
    save_memory_to_json(memory)

    # 대화 내역 갱신
    st.session_state.messages.append({"role": "assistant", "content": response})
