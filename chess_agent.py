import streamlit as st
import chess
import chess.svg
import json
from langchain_openai import ChatOpenAI
import openai

# 🔥 Streamlit 페이지 설정
st.set_page_config(layout="wide")
st.sidebar.markdown("### 🔑 OpenAI API Key 입력")
api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요:", type="password")

if api_key:
    st.session_state["api_key"] = api_key

# 🔥 체스 보드와 채팅 기록 초기화
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔥 LLM 에이전트 초기화 (API 키 필요)
if "api_key" in st.session_state and st.session_state["api_key"]:
    llm_agent = ChatOpenAI(
        model_name="gpt-4o",
        max_tokens=200,
        temperature=0.7,
        api_key=st.session_state["api_key"]
    )
else:
    llm_agent = None

# ============================================
# 1. 체스 라이브러리 기능을 도구(tool)로 래핑
# ============================================

def board_state_tool(board: chess.Board):
    """현재 체스 보드의 FEN 상태를 반환합니다."""
    return f"현재 보드 상태 (FEN): {board.fen()}"

def legal_moves_tool(board: chess.Board):
    """현재 보드에서 가능한 모든 합법 수(SAN 형식)를 반환합니다."""
    moves = [board.san(move) for move in board.legal_moves]
    return f"합법 수: {', '.join(moves)}"

def move_execution_tool(move_str: str, board: chess.Board):
    """SAN 혹은 UCI 형식의 이동 명령을 검증하여 실행합니다."""
    try:
        try:
            move = board.parse_san(move_str)
        except Exception:
            move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            board.push(move)
            return f"수 '{move_str}' 이동을 성공적으로 수행함. 새로운 FEN: {board.fen()}"
        else:
            return f"'{move_str}'은(는) 합법 수가 아님."
    except Exception as e:
        return f"이동 수행 중 오류 발생: {e}"

def piece_info_tool(square: str, board: chess.Board):
    """특정 칸(예: 'e4')에 있는 기물 정보를 반환합니다."""
    try:
        square_index = chess.parse_square(square)
        piece = board.piece_at(square_index)
        if piece:
            return f"{square}에 있는 기물: {piece.symbol()}"
        else:
            return f"{square}에는 기물이 없음."
    except Exception as e:
        return f"기물 정보 확인 중 오류 발생: {e}"

def search_chess_knowledge_tool(query: str):
    """
    외부 체스 지식 베이스(예: 게임 데이터베이스, 전략 정보)를 검색합니다.
    (여기서는 시뮬레이션 결과를 반환합니다.)
    """
    return f"'{query}'에 대한 체스 지식 검색 결과: (시뮬레이션 결과)"

def set_board_state_tool(fen_str: str, board: chess.Board):
    """주어진 FEN 정보를 사용하여 보드 상태를 업데이트합니다."""
    try:
        new_board = chess.Board(fen_str)
        st.session_state.board = new_board
        return f"보드 상태를 업데이트함. 새로운 FEN: {new_board.fen()}"
    except Exception as e:
        return f"보드 상태 업데이트 중 오류 발생: {e}"

# 도구 이름과 함수 매핑
tools = {
    "BoardStateTool": board_state_tool,
    "LegalMovesTool": legal_moves_tool,
    "MoveExecutionTool": move_execution_tool,
    "PieceInfoTool": piece_info_tool,
    "SearchChessKnowledgeTool": search_chess_knowledge_tool,
    "SetBoardStateTool": set_board_state_tool
}

# ============================================
# 2. 에이전트 함수: 자연어 명령에 따라 적절한 도구 호출
# ============================================

def agent_decision(user_query: str, board: chess.Board):
    """
    사용자의 자연어 입력을 분석하여 어떤 도구를 호출할지 결정합니다.
    출력은 반드시 JSON 형식이어야 하며, 키 "tool"과 "args"를 포함해야 합니다.
    모든 응답은 한국어로 작성하고, 사용자의 말투(반말/존댓말)를 최대한 따라해.
    
    예시: {"tool": "MoveExecutionTool", "args": {"move": "e2e4"}}
    만약 도구 호출이 필요 없으면, {"tool": "None", "args": {}}라고 응답해.
    
    추가사항:
    - 사용자가 FEN 정보를 포함한 입력이면 SetBoardStateTool을 사용.
    - "수행해줘","이동해줘" 등의 문구가 포함되어 있으면 반드시 이동 실행(MoveExecutionTool)을 호출하고,
      그 결과를 반영하도록 해.
    - 자연어 이동 지시가 들어오면 이를 유효한 체스 이동 형식으로 변환해서 MoveExecutionTool을 호출해.
    """
    tool_descriptions = """
BoardStateTool: 현재 보드의 FEN 상태를 반환.
LegalMovesTool: 현재 보드에서 가능한 합법 수(SAN 형식)를 반환.
MoveExecutionTool: SAN 또는 UCI 형식의 이동 명령을 실행하여 보드를 업데이트.
PieceInfoTool: 특정 칸(예: 'e4')에 있는 기물 정보를 반환.
SearchChessKnowledgeTool: 외부 체스 지식 베이스를 검색하여 전략이나 역사적 게임 정보를 제공.
SetBoardStateTool: 주어진 FEN 정보를 사용하여 보드 상태를 업데이트.
    """
    prompt = f"""
너는 체스 어시스턴트야. 다음 도구들을 사용할 수 있어:
{tool_descriptions}

현재 보드 상태 (FEN): {board.fen()}

사용자 입력: "{user_query}"

위 입력에 대해, 어떤 도구를 사용해야 하는지 판단하고, 그 도구와 필요한 인자들을 JSON 형식으로 응답해줘.
출력은 반드시 JSON 형식이어야 하며, 예시는 다음과 같아:
{{"tool": "MoveExecutionTool", "args": {{"move": "e2e4"}}}}
만약 도구 호출이 필요 없으면, {{"tool": "None", "args": {{}}}}라고 응답해.
단, 모든 응답은 한국어로 작성하고, 사용자의 말투(반말/존댓말)를 따라해.
또한, 만약 입력에 FEN 정보가 포함되어 있다면 SetBoardStateTool을 사용하고,
"수행","이동" 이라는 문구가 있거나 SAN형식의 이동이 입력으로 들어오면 반드시 이동 실행을 포함하여 MoveExecutionTool을 호출해서 이동된 보드를 반환해줘.
    """
    response = llm_agent.invoke(prompt)
    try:
        decision = json.loads(response.content)
    except Exception as e:
        decision = {"tool": "None", "args": {}}
    return decision

def agent_final_response(user_query: str, tool_results: dict, board: chess.Board):
    """
    도구 호출 결과와 현재 보드 상태를 바탕으로 최종 자연어 답변을 생성합니다.
    모든 답변은 한국어로 작성하며, 사용자의 말투를 최대한 따라해.
    만약 도구 결과 중 MoveExecutionTool 호출 결과가 있다면, 최종 답변에 이동 수행 결과("이걸 수행했어" 등)를 반드시 포함해줘.
    """
    results_str = "\n".join([f"{tool}: {result}" for tool, result in tool_results.items()])
    prompt = f"""
너는 체스 전문가야. 아래 도구 호출 결과와 현재 보드 상태를 참고하여 사용자에게 최종 답변을 해줘.
사용자 입력: "{user_query}"

도구 결과:
{results_str}

현재 보드 상태: {board.fen()}

만약 도구 결과 중 MoveExecutionTool 호출 결과가 있다면, 그 결과를 반영하여 "이걸 수행했어"라는 표현을 포함해서 답변해줘.
위 정보를 바탕으로, 간결하고 도움이 되는 답변을 한국어로 작성해줘.
사용자의 말투(반말/존댓말)를 최대한 따라해.
    """
    final_response = llm_agent.invoke(prompt)
    return final_response.content

def agent_process(user_query: str, board: chess.Board):
    """
    전체 에이전트 파이프라인:
      1. 사용자 입력을 분석하여 호출할 도구 결정 (agent_decision)
      2. 선택한 도구를 실행하고 결과를 수집
      3. 도구 결과와 보드 상태를 바탕으로 최종 답변 생성 (agent_final_response)
    """
    if llm_agent is None:
        return "API 키가 필요합니다. 사이드바에 입력하세요."
    
    decision = agent_decision(user_query, board)
    tool_results = {}
    selected_tool = decision.get("tool", "None")
    args = decision.get("args", {})
    
    if selected_tool != "None" and selected_tool in tools:
        if selected_tool == "MoveExecutionTool":
            move_arg = args.get("move", "")
            # 실제 이동을 실행하여 보드를 업데이트함
            result = tools[selected_tool](move_arg, board)
            tool_results[selected_tool] = result
        elif selected_tool == "PieceInfoTool":
            square_arg = args.get("square", "")
            result = tools[selected_tool](square_arg, board)
            tool_results[selected_tool] = result
        elif selected_tool == "SearchChessKnowledgeTool":
            query_arg = args.get("query", user_query)
            result = tools[selected_tool](query_arg)
            tool_results[selected_tool] = result
        elif selected_tool == "SetBoardStateTool":
            fen_arg = args.get("fen", "")
            result = tools[selected_tool](fen_arg, board)
            tool_results[selected_tool] = result
            board = st.session_state.board  # 업데이트된 보드 반영
        else:
            # BoardStateTool 또는 LegalMovesTool 등
            result = tools[selected_tool](board)
            tool_results[selected_tool] = result
    else:
        tool_results["None"] = "도구 호출이 필요하지 않음."
    
    final_answer = agent_final_response(user_query, tool_results, board)
    return final_answer

# ============================================
# 3. UI 레이아웃: 체스 보드와 통합 자연어 인터페이스
# ============================================

col1, col2 = st.columns([1.5, 1])

# --- 왼쪽: 체스 보드 출력 ---
with col1:
    st.markdown("### 체스 보드")
    board_placeholder = st.empty()

    def update_board():
        svg = chess.svg.board(board=st.session_state.board, size=350)
        board_placeholder.markdown(
            f'<div style="width: 350px; height: 350px;">{svg}</div>',
            unsafe_allow_html=True
        )

    update_board()

# --- 오른쪽: 통합 자연어 인터페이스 (에이전트) ---
with col2:
    st.markdown("### 체스 에이전트")
    user_input = st.text_input(
        "명령을 입력하세요: "
    )
    
    if st.button("전송"):
        # ▶ 전처리: 입력이 "FEN:"으로 시작하면 바로 보드 업데이트 수행
        if user_input.strip().upper().startswith("FEN:"):
            fen_str = user_input.strip()[4:].strip()  # "FEN:" 제거 후 FEN 문자열 추출
            result = set_board_state_tool(fen_str, st.session_state.board)
            st.session_state.chat_history.insert(0, f"🤖 에이전트: {result}")
            st.session_state.chat_history.insert(0, f"👤 사용자: {user_input}")
            update_board()
        else:
            response = agent_process(user_input, st.session_state.board)
            st.session_state.chat_history.insert(0, f"🤖 에이전트: {response}")
            st.session_state.chat_history.insert(0, f"👤 사용자: {user_input}")
            update_board()

    st.markdown("### 채팅 기록")
    for msg in st.session_state.chat_history:
        st.write(msg)
