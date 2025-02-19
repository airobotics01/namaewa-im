import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

from deep_translator import GoogleTranslator
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens = 150)

# 감정 분석기 초기화
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# 🔹 번역 툴
@tool
def translate_tool(text: str) -> str:
    """주어진 텍스트를 영어로 번역하는 툴"""
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        return f"🔠 번역 결과: {translated_text}"
    except Exception as e:
        return f"❌ 번역 오류: {str(e)}"

# 🔹 요약 툴
@tool
def summarize_tool(text: str) -> str:
    """주어진 텍스트를 요약하는 툴"""
    try:
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"다음 내용을 짧게 요약해줘: {text}"}
        ])

        return f"📄 요약 결과: {response["output"]}"  # ✅ 올바른 방식
    except Exception as e:
        return f"❌ 요약 오류: {str(e)}"


# 🔹 감정 분석 툴
@tool
def analyze_sentiment(text: str) -> str:
    """한국어 텍스트를 영어로 번역한 후 감정 분석을 수행하는 툴"""
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        sentiment_scores = sia.polarity_scores(translated_text)
        compound_score = sentiment_scores["compound"]
        sentiment_label = "긍정적 😀" if compound_score > 0.05 else "부정적 😞" if compound_score < -0.05 else "중립적 😐"
        return f"📊 감정 분석 결과: {sentiment_label} (점수: {compound_score})"
    except Exception as e:
        return f"❌ 감정 분석 오류: {str(e)}"

# 🔹 자동 웹 검색 툴 (DuckDuckGo 사용)
search_tool = DuckDuckGoSearchRun()

# 🔹 문서 요약 툴 (검색 결과를 요약)
@tool
def document_reader_tool(query: str) -> str:
    """웹에서 검색한 정보를 읽고 요약하는 툴"""
    try:
        search_results = search_tool.run(query)
        
        if not search_results:
            return "❌ 검색 결과를 찾을 수 없습니다."

        # 최신 OpenAI API 적용 (llm.invoke() 사용)
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"다음 검색 결과를 종합하여 답변을 생성해줘: {search_results}"}
        ])

        return f"🌐 검색 결과 요약: {response.content}"
    
    except Exception as e:
        return f"❌ 문서 읽기 오류: {str(e)}"

# 🔹 Python 코드 실행 툴 추가
python_repl_tool = PythonREPLTool()

# 🔹 에이전트 초기화 (LLM + Tools)
tools = [translate_tool, summarize_tool, analyze_sentiment, document_reader_tool, python_repl_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # LLM이 Task를 분석하여 적절한 Tool을 선택
    verbose=True
)

# 🔹 에이전트 실행
def run_agent():
    while True:
        user_input = input("\n사용자 입력: ").strip()
        if user_input.lower() in ["그만", "종료", "quit"]:
            print("🛑 대화 종료")
            break
        
        response = agent.invoke(user_input)
        print(f"🤖: {response["output"]}")


def test_tools():
    print("\n🔹 [TEST] 번역 툴")
    translation_result = translate_tool("안녕하세요, 반갑습니다!")
    print(f"✅ 번역 결과: {translation_result}\n")

    print("\n🔹 [TEST] 요약 툴")
    summary_result = summarize_tool("LangChain은 다양한 AI 모델을 연결하여 자동화된 시스템을 구축하는 프레임워크입니다.")
    print(f"✅ 요약 결과: {summary_result}\n")

    print("\n🔹 [TEST] 감정 분석 툴")
    sentiment_result = analyze_sentiment("오늘 하루 너무 힘들었어. 정말 최악이야.")
    print(f"✅ 감정 분석 결과: {sentiment_result}\n")

    print("\n🔹 [TEST] 웹 검색 및 문서 요약 툴")
    document_summary_result = document_reader_tool("2024년 AI 최신 연구 동향")
    print(f"✅ 웹 검색 및 문서 요약 결과: {document_summary_result}\n")

    print("\n🔹 [TEST] Python 코드 실행 툴")
    python_code = "print(sum([i for i in range(10)]))"  # 0~9까지의 합 (45)
    python_result = python_repl_tool.run(python_code)
    print(f"✅ Python 코드 실행 결과: {python_result}\n")

# 실행
if __name__ == "__main__":
    # test_tools()
    run_agent()

