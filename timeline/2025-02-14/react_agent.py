import os
import openai
import threading
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# ✅ 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ LLM 설정 (GPT-4o 사용)
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=150)

# ✅ 글로벌 저장소 (각 닉네임별 저장)
store = InMemoryStore()

# ✅ 유저별 체크포인트 & 메모리 저장소 (스레드별 저장)
user_checkpoints = {}
user_memories = {}

# ✅ JSON 파일 경로
MEMORY_FILE = "./chat_memory.json"

# ✅ 대화 기록을 JSON으로 저장하는 함수
def serialize_messages(messages):
    """메시지 객체를 JSON 직렬화 가능한 형태로 변환"""
    return [{"type": type(msg).__name__, "content": msg.content} for msg in messages]

def deserialize_messages(messages):
    """JSON에서 불러온 데이터를 다시 메시지 객체로 변환"""
    msg_type_map = {"HumanMessage": HumanMessage, "AIMessage": AIMessage, "SystemMessage": SystemMessage}
    return [msg_type_map[msg["type"]](content=msg["content"]) for msg in messages]

def save_data():
    """현재 모든 유저의 대화 메모리를 JSON 파일로 저장 (실시간 저장)"""
    data = {}

    for user, memory in user_memories.items():
        messages = memory.chat_memory.messages
        if messages:  # 🔹 비어있는 데이터는 저장하지 않음
            data[user] = serialize_messages(messages)

    if not data:
        print("🚫 저장할 대화 데이터가 없습니다.")
        return

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"💾 ✅ 대화 기록이 저장되었습니다: {MEMORY_FILE}")

    # 파일 저장 후 확인
    if os.path.exists(MEMORY_FILE):
        print("✅ JSON 파일이 정상적으로 생성됨!")
    else:
        print("❌ JSON 파일이 생성되지 않음!")



# ✅ JSON에서 대화 기록 불러오는 함수
def load_data():
    """저장된 JSON 파일에서 유저별 대화 메모리를 불러옴"""
    global user_memories
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print("🚫 저장된 대화가 없습니다. 새로 시작합니다.")
                return
            data = json.loads(content)

        for user, messages in data.items():
            user_memories[user] = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="output"
            )
            user_memories[user].chat_memory.messages = deserialize_messages(messages)
        print(f"💾 🔄 대화 기록이 복원되었습니다: {MEMORY_FILE}")
    
    except (FileNotFoundError, json.JSONDecodeError):
        print("🚫 저장된 대화 없음. 새로 시작합니다.")
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)  # 🔹 빈 JSON 파일 생성


# ✅ 프로그램 시작 시 데이터 로드
load_data()

# ✅ 검색 툴 (실시간 정보 제공)
search_tool = DuckDuckGoSearchRun()

# 🔹 **대화 기억 툴 (자연스러운 응답)**
@tool
def memory_tool(query: str) -> str:
    """사용자의 이전 대화를 기억하고 자연스럽게 이어서 대화하는 툴"""
    thread_id = threading.current_thread().name  # 현재 스레드(닉네임) 가져오기
    memory = user_memories.get(thread_id)  # 해당 유저의 메모리 가져오기

    if memory:
        messages = memory.chat_memory.messages  # ✅ 저장된 대화 가져오기
        for msg in reversed(messages):
            if query.lower() in msg.content.lower():
                return msg.content  # ✅ 자연스럽게 해당 내용을 대화에 녹여서 사용
    return ""

# 🔹 **검색 툴 (웹에서 정보 검색 후 요약)**
@tool
def search_summary_tool(query: str) -> str:
    """웹 검색을 통해 필요한 정보를 제공하는 툴"""
    search_results = search_tool.run(query)
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant that summarizes search results."},
        {"role": "user", "content": f"다음 검색 결과를 요약해줘: {search_results}"}
    ])
    return response.content  # ✅ 불필요한 라벨 제거

# ✅ 사용 가능한 툴 목록
tools = [memory_tool, search_summary_tool]

# ✅ 실시간 스트리밍 출력 함수 (대화 내용을 기억)
def print_stream(graph, inputs, config):
    """체크포인트와 메모리를 활용하여 지속적인 대화를 유지"""
    for s in graph.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# ✅ 챗봇 실행 함수 (닉네임별 대화 기억)
def run_agent():
    thread_id = input("\n✅ 당신의 닉네임을 입력하세요: ").strip()

    # ✅ 닉네임별 개별 체크포인트 & 메모리 생성
    if thread_id not in user_checkpoints:
        user_checkpoints[thread_id] = MemorySaver()
        user_memories.setdefault(thread_id, ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="output"
        ))

    checkpointer = user_checkpoints[thread_id]
    memory = user_memories[thread_id]

    # ✅ 닉네임별 개별 graph 생성
    graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer, store=store)

    # ✅ 현재 스레드(사용자 닉네임) 저장
    threading.current_thread().name = thread_id

    config = {"configurable": {"thread_id": thread_id}}
    graph.get_state(config)

    while True:
        user_input = input("\n사용자 입력: ").strip()

        # ✅ "커널 종료" 입력 시 프로그램 종료
        if user_input.lower() in ["커널 종료", "kernel shutdown"]:
            print("🛑 커널 종료 요청됨. 대화 기록을 저장하고 종료합니다.")
            save_data()
            os._exit(0)  # 프로그램 강제 종료

        # ✅ "그만", "종료", "quit" 입력 시 JSON 파일 실시간 저장 후 종료
        elif user_input.lower() in ["그만", "종료", "quit"]:
            print("🛑 대화 종료")
            save_data()  # 🔹 실시간 저장
            os._exit(0)

        # ✅ LLM 실행 후 응답 저장
        inputs = {"messages": [("user", user_input)]}
        response = graph.invoke(inputs, config=config)

        # ✅ 유저별 LangChain 메모리에 대화 저장
        memory.save_context({"input": user_input}, {"output": response["messages"][-1].content})

        # ✅ 대화가 끝날 때마다 JSON 파일 자동 업데이트 (실시간 저장)
        save_data()

        print_stream(graph, inputs, config)

# ✅ 실행 (멀티스레드 지원)
if __name__ == "__main__":
    while True:
        thread = threading.Thread(target=run_agent)
        thread.start()
        thread.join()
