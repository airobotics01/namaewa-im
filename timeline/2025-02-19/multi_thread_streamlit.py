import os
import json
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
import random

# ✅ 환경 변수 로드
load_dotenv()

# ✅ JSON 파일에서 대화 데이터를 불러오는 함수 (사용자별 Thread 관리)
def load_json_to_memory(memory, thread_id, filename):
    """JSON 파일에서 특정 사용자의 대화 내역을 불러와 ConversationBufferMemory에 저장"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if thread_id not in data:
            print(f"사용자 {thread_id}에 대한 대화 기록이 없습니다.")
            return

        for conversation in data[thread_id]:
            user_input = conversation.get("user", "")
            agent_response = conversation.get("agent", "")
            if user_input and agent_response:
                memory.save_context({"input": user_input}, {"output": agent_response})
        print(f"✅ {filename}에서 사용자 {thread_id}의 대화 내역이 메모리에 저장되었습니다.")
    except FileNotFoundError:
        print(f"❌ {filename} 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        print(f"❌ {filename} 파일이 올바른 JSON 형식이 아닙니다.")

# ✅ JSON 파일에 대화 내역 저장 함수
def save_memory_to_json(user_input, agent_response, thread_id, filename):
    """사용자의 대화 내역을 JSON 파일에 추가 저장하는 함수"""

    # 기존 데이터 로드 (파일이 존재하면 읽고, 없으면 빈 딕셔너리 생성)
    data = {}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    # 해당 사용자(thread_id)의 대화 기록이 없으면 새로 생성
    if thread_id not in data:
        data[thread_id] = []

    # 새로운 대화 추가 (중복 방지)
    new_entry = {"user": user_input, "agent": agent_response}
    if new_entry not in data[thread_id]:  # 중복 방지
        data[thread_id].append(new_entry)

    # JSON 파일에 저장
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ 사용자 '{thread_id}'의 대화 내역이 {filename}에 저장되었습니다.")


# ✅ Streamlit 페이지 설정
st.set_page_config(page_title="🤖 AI 챗봇", layout="wide")

# ✅ 사이드바 설정
with st.sidebar:
    st.title("🔑 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    thread_id = st.text_input("User ID")
    filename = st.text_input("Create file name")
    filename = f'{filename}.json'

if not openai_api_key or not thread_id or not filename:
    st.warning("⚠️ OpenAI API Key와 ID와 대화를 저장할 파일명을 입력하세요!")
    st.stop()



# ✅ LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=200, api_key=openai_api_key)

# ✅ 체크포인트 저장소
store = InMemoryStore()
checkpointer = MemorySaver()

# ✅ LangChain Memory 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")
load_json_to_memory(memory, thread_id, filename=filename)  # JSON에서 대화 불러오기

# ✅ 검색 툴
search_tool = DuckDuckGoSearchRun()

# @tool
# def memory_tool(query: str) -> str:
#     """사용자의 이전 대화를 기억하고 자연스럽게 이어서 대화하는 툴"""
#     messages = memory.load_memory_variables({}).get("chat_history", "")
#     return "\n".join(messages) if messages else "대화 기록이 없습니다."

# @tool
# def memory_tool(query: str) -> str:
#     """사용자의 이전 대화를 기억하고 자연스럽게 이어서 대화하는 툴"""
#     messages = memory.load_memory_variables({}).get("chat_history", [])
    
#     # 메시지 객체를 문자열로 변환
#     messages_str = [msg.content for msg in messages]  # ✅ HumanMessage, AIMessage 객체를 텍스트로 변환

#     return "\n".join(messages_str) if messages_str else "대화 기록이 없습니다."

@tool
def memory_tool(query: str) -> dict:
    """사용자의 이전 대화를 JSON 형식으로 반환하여 AI가 쉽게 이해하도록 함"""
    messages = memory.load_memory_variables({}).get("chat_history", [])
    messages_data = [{"role": msg.type, "content": msg.content} for msg in messages]

    return {"history": messages_data} if messages_data else {"history": []}


@tool
def search_summary_tool(query: str) -> str:
    """웹 검색을 통해 필요한 정보를 제공하는 툴"""
    search_results = search_tool.run(query)
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant that summarizes search results."},
        {"role": "user", "content": f"다음 검색 결과를 요약해줘: {search_results}"}
    ])
    return response.content

@tool
def emotional_response_tool(user_input: str) -> str:
    """사용자의 감정에 따라 자연스럽게 반응하는 툴"""
    if any(word in user_input.lower() for word in ["힘들어", "우울해", "슬퍼", "지쳤어", "짜증나"]):
        return random.choice(["괜찮아? 무슨 일 있었어?", "음… 나한테 말해도 괜찮아. 무슨 일인데?"])
    elif any(word in user_input.lower() for word in ["기뻐", "좋아", "행복해", "신나"]):
        return random.choice(["오! 좋은 일이 있었구나!", "와, 너 정말 행복해 보인다!"])
    return ""

# ✅ ReAct 기반 챗봇 생성
tools=[memory_tool, search_summary_tool, emotional_response_tool]

# agent = create_react_agent(llm, tools = tools, 
#                            checkpointer=checkpointer, store=store,
#                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # LLM이 Task를 분석하여 적절한 Tool을 선택
#                            verbose=True)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,  
    handle_parsing_errors=True,
)


# ✅ 실시간 스트리밍 출력 함수 (대화 내용을 기억)
def print_stream(graph, inputs, config):
    """체크포인트와 메모리를 활용하여 지속적인 대화를 유지"""
    for s in graph.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]  
        if isinstance(message, tuple):
            print(message)  
        else:
            message.pretty_print()  

# ✅ Streamlit UI 생성
st.title("🤖 AI 챗봇")

# ✅ Streamlit의 세션 상태에서 thread_id 별로 대화 내역 관리
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}  # 모든 thread_id의 대화 저장

# ✅ 사용자가 새로운 User ID를 입력했을 때 처리
if thread_id != st.session_state.get("thread_id", None):
    # ✅ 기존 ID의 대화 내역 저장
    if st.session_state.get("thread_id") is not None:
        st.session_state["chat_history"][st.session_state["thread_id"]] = st.session_state["messages"]

    # ✅ 새 thread_id로 변경
    st.session_state["thread_id"] = thread_id

    # ✅ 새 ID의 대화 내역을 불러오거나 초기화
    if thread_id in st.session_state["chat_history"]:
        st.session_state["messages"] = st.session_state["chat_history"][thread_id]
    else:
        st.session_state["messages"] = []  # 새로운 ID는 빈 대화창
        load_json_to_memory(memory, thread_id, filename=filename)  # JSON에서 불러오기

# ✅ 기존 채팅 내역 출력
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    inputs = {"input": user_input}
    config = {"configurable": {"thread_id": thread_id}}

    # ✅ LangChain Agent 실행
    response = agent.invoke(inputs, config=config)
    ai_response = response.get("output", "응답을 가져올 수 없습니다.")

    # ✅ Memory에 대화 저장
    memory.save_context({"input": user_input}, {"output": ai_response})
    save_memory_to_json(user_input, ai_response, thread_id, filename=filename)

    # ✅ 세션 상태에 메시지 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state["messages"].append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)

    # ✅ 현재 thread_id의 대화 내역을 chat_history에 저장
    st.session_state["chat_history"][thread_id] = st.session_state["messages"]

    # ✅ 채팅창을 새로고침하여 반영
    st.rerun()
