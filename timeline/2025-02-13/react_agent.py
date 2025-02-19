import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# ✅ 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ LLM 설정 (GPT-4o 사용)
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens= 150)

# ✅ 체크포인트 저장소 (대화 상태 저장)
store = InMemoryStore()
checkpointer = MemorySaver()

# ✅ 대화 기억을 위한 메모리
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")

# ✅ 검색 툴 (실시간 정보 제공)
search_tool = DuckDuckGoSearchRun()

# 🔹 **대화 기억 툴 (자연스러운 응답)**
@tool
def memory_tool(query: str) -> str:
    """사용자의 이전 대화를 기억하고 자연스럽게 이어서 대화하는 툴"""
    messages = memory.chat_memory.messages  # ✅ 저장된 대화 가져오기
    for msg in reversed(messages):
        if query.lower() in msg.content.lower():
            return msg.content  # ✅ 그냥 자연스럽게 해당 내용을 대화에 녹여서 사용
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

# 🔹 **위로 & 감정 반응 툴 (자연스럽게 말하기)**
import random
@tool
def emotional_response_tool(user_input: str) -> str:
    """사용자의 감정에 따라 자연스럽게 반응하는 툴"""
    
    # 부정적인 감정 (힘들거나 우울할 때)
    if any(word in user_input.lower() for word in ["힘들어", "우울해", "슬퍼", "지쳤어", "짜증나", "속상해"]):
        responses = [
            "오늘 무슨 일 있었어? 얘기해 줄래?",
            "괜찮아? 무슨 일 때문에 그런 기분이 들었어?",
            "음… 나한테 말해도 괜찮아. 무슨 일인데?",
            "그랬구나... 나도 그런 기분 들 때가 있어. 조금이라도 편해질 수 있게 같이 이야기할까?"
        ]
        return random.choice(responses)
    
    # 긍정적인 감정 (행복하거나 신났을 때)
    elif any(word in user_input.lower() for word in ["기뻐", "좋아", "행복해", "신나", "설레", "즐거워"]):
        responses = [
            "오! 좋은 일이 있었구나! 나도 궁금한데, 무슨 일이야?",
            "그거 진짜 좋겠다! 어떤 일인데?",
            "들으니까 나도 기분이 좋아지네! 더 얘기해줄래?",
            "와, 너 정말 행복해 보인다! 무슨 일인지 알려줘!"
        ]
        return random.choice(responses)

    # 화났거나 짜증날 때
    elif any(word in user_input.lower() for word in ["화나", "빡쳐", "짜증", "열받아", "답답해"]):
        responses = [
            "그랬어? 많이 속상했겠다. 무슨 일 있었어?",
            "아... 그런 상황이면 진짜 답답할 것 같아. 좀 더 얘기해 줄래?",
            "너무 화나면 진짜 힘들지... 나한테라도 이야기해봐.",
            "괜찮아, 천천히 말해도 돼. 어떤 일이야?"
        ]
        return random.choice(responses)

    # 걱정하거나 고민이 있을 때
    elif any(word in user_input.lower() for word in ["걱정", "고민", "불안", "걱정돼", "어떡하지"]):
        responses = [
            "뭔가 걱정되는 게 있구나. 어떤 일이야?",
            "괜찮아, 천천히 말해도 돼. 혹시 내가 도와줄 수 있을까?",
            "그런 고민이 있으면 정말 신경 쓰이겠네. 나랑 이야기해볼래?",
            "음... 그러면 이런 방법은 어때? 같이 한번 생각해보자!"
        ]
        return random.choice(responses)

    return ""

# ✅ 사용 가능한 툴 목록
tools = [memory_tool, search_summary_tool, emotional_response_tool]

# ✅ ReAct 기반 챗봇 생성 (대화 기억 + 검색 + 감정 반응)
graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer, store=store)

# ✅ 실시간 스트리밍 출력 함수 (대화 내용을 기억)
def print_stream(graph, inputs, config):
    """체크포인트와 메모리를 활용하여 지속적인 대화를 유지"""
    for s in graph.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]  
        if isinstance(message, tuple):
            print(message)  
        else:
            message.pretty_print()  

# ✅ 챗봇 실행 함수
def run_agent():
    thread_id = input("\n✅당신의 닉네임을 입력하세요: ").strip()
    config = {"configurable": {"thread_id": thread_id}}
    graph.get_state(config)

    while True:
        user_input = input("\n사용자 입력: ").strip()
        if user_input.lower() in ["그만", "종료", "quit"]:
            print("🛑 대화 종료")
            break

        # ✅ LLM 실행 후 응답 저장
        inputs = {"messages": [("user", user_input)]}
        response = graph.invoke(inputs, config=config)

        # ✅ LangChain 메모리에 사용자 입력 및 AI 응답 저장
        memory.save_context({"input": user_input}, {"output": response["messages"][-1].content})

        print_stream(graph, inputs, config)

# ✅ 실행
if __name__ == "__main__":
    run_agent()
