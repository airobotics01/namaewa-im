import os
import openai
import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import random

# ✅ Streamlit 페이지 설정
st.set_page_config(page_title="🤖 AI 챗봇", layout="wide")

# ✅ 왼쪽 사이드바 (API Key & Thread ID 입력)
with st.sidebar:
    st.title("🔑 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    thread_id = st.text_input("🆔 대화 Thread ID (예: user1)")

# ✅ 경고 메시지 출력
if not openai_api_key or not thread_id:
    st.warning("⚠️ OpenAI API Key와 Thread ID를 입력하세요!")
    st.stop()

# ✅ LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=150, api_key=openai_api_key)

# ✅ 체크포인트 저장소 (대화 상태 저장)
store = InMemoryStore()
checkpointer = MemorySaver()

# ✅ **LangChain 대화 메모리**
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")

# ✅ 검색 툴 (실시간 정보 제공)
search_tool = DuckDuckGoSearchRun()

# 🔹 **대화 기억 툴 (이전 대화 활용)**
@tool
def memory_tool(query: str) -> str:
    """사용자의 이전 대화를 기억하고 자연스럽게 이어서 대화하는 툴"""
    messages = memory.load_memory_variables({})["chat_history"]
    return "\n".join([f"{msg.role}: {msg.content}" for msg in messages]) if messages else "대화 기록이 없습니다."

# 🔹 **검색 툴**
@tool
def search_summary_tool(query: str) -> str:
    """웹 검색을 통해 필요한 정보를 제공하는 툴"""
    search_results = search_tool.run(query)
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant that summarizes search results."},
        {"role": "user", "content": f"다음 검색 결과를 요약해줘: {search_results}"}
    ])
    return response.content

# 🔹 **감정 반응 툴**
@tool
def emotional_response_tool(user_input: str) -> str:
    """사용자의 감정에 따라 자연스럽게 반응하는 툴"""
    if any(word in user_input.lower() for word in ["힘들어", "우울해", "슬퍼", "지쳤어", "짜증나", "속상해"]):
        return random.choice(["괜찮아? 무슨 일 있었어?", "음… 나한테 말해도 괜찮아. 무슨 일인데?", "그랬구나... 나도 그런 기분 들 때가 있어."])
    elif any(word in user_input.lower() for word in ["기뻐", "좋아", "행복해", "신나", "설레", "즐거워"]):
        return random.choice(["오! 좋은 일이 있었구나! 무슨 일이야?", "와, 너 정말 행복해 보인다!"])
    return ""

# ✅ 사용 가능한 툴 목록
tools = [memory_tool, search_summary_tool, emotional_response_tool]

# ✅ **ReAct 기반 챗봇 생성 (LangChain Memory 적용)**
graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer, store=store)

# ✅ **Streamlit 세션 상태 저장 (이전 대화 유지)**
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ **채팅 UI (항상 보이도록 유지)**
st.title("🤖 AI 챗봇")
st.markdown("💬 **OpenAI 기반 AI 챗봇과 대화하세요!**")

# ✅ **기존 채팅 메시지 출력**
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ **사용자 입력 받기**
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    # ✅ **이전 대화 기록 불러오기**
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # ✅ **LLM 실행을 위한 메시지 구성**
    messages = chat_history + [("user", user_input)]
    inputs = {"messages": messages}
    config = {"configurable": {"thread_id": thread_id}}

    # ✅ **LangGraph 실행**
    response = graph.invoke(inputs, config=config)
    ai_response = response["messages"][-1].content

    # ✅ **LangChain Memory에 대화 저장**
    memory.save_context({"input": user_input}, {"output": ai_response})

    # ✅ **Streamlit 세션에도 저장**
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
