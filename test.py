import sys
import asyncio
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from agent import get_agent
from langchain_core.messages import SystemMessage, HumanMessage
from database import init_db

init_db()

agent = get_agent("gemini-3.5-flash")


config = {
    "configurable": {
        "thread_id": "test_thread_id",
    }
}

async def run_test():
    async for message_chunk, metadata in agent.astream(
        {'messages': [HumanMessage(content="What is My name?")]},
        config=config,
        stream_mode='messages'
    ):
        if message_chunk.content:
            print(message_chunk.content, end=" ", flush=True)

if __name__ == "__main__":
    asyncio.run(run_test())
    