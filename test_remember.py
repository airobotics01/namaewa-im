'''

코드 실행 흐름
JSON 파일을 열어 사용자 대화 데이터를 불러옴

"user1" 또는 "user2"를 지정하여 특정 사용자의 대화를 로드할 수 있음.
ConversationBufferMemory에 대화 저장

memory.save_context()를 사용하여 JSON의 "user"와 "agent" 데이터를 저장.

'''

import json
from langchain.memory import ConversationBufferMemory

# ✅ JSON 파일에서 대화 데이터를 불러오는 함수
def load_json_to_memory(memory, thread_id, filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        if thread_id not in data:
            print(f"사용자 {thread_id}에 대한 대화 기록이 없습니다.")
            return

        # JSON 데이터를 ConversationBufferMemory에 삽입
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

# ✅ ConversationBufferMemory 객체 생성
memory = ConversationBufferMemory()

# ✅ JSON 데이터를 불러와 메모리에 저장
load_json_to_memory(memory, filename="chat_memory.json", thread_id="user1")

# ✅ 저장된 대화 내역 확인
print(memory.load_memory_variables({}))


