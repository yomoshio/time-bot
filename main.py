# main.py
import asyncio
from datetime import datetime
from langgraph.graph import MessageGraph, ToolExecutor
from langgraph.checkpoint import NoOpCheckpoint
from langgraph.prebuilt import chat_agent_executor
from aiohttp import ClientSession
from typing import List, Dict


OPENAI_API_KEY = "your-api-key"
PROXY = "http://your-proxy:port"


class OpenAIApi:
    def __init__(self, key: str, proxy: str) -> None:
        self.key = key
        self.headers = {'Authorization': f'Bearer {key}'}
        self.proxy = proxy

    async def ask(self, session: ClientSession, messages: List[Dict], model: str, timeout=30):
        url = 'https://api.openai.com/v1/chat/completions'
        data = {'model': model, 'messages': messages}
        response = await session.post(url=url, json=data, headers=self.headers, proxy=self.proxy, timeout=timeout)
        if response.status != 200:
            raise Exception(f'Ошибка ChatGPT ({response.status}):\n{await response.text()}')
        result = await response.json()
        return result['choices'][0]['message']



def get_current_time() -> dict:
    return {"utc": datetime.utcnow().isoformat() + "Z"}



tools = {
    "get_current_time": lambda _: get_current_time()
}


async def run_chat():
    key = OPENAI_API_KEY
    proxy = PROXY

    openai_api = OpenAIApi(key, proxy)

    async with ClientSession() as session:
        async def llm_function(messages: List[Dict]):
            result = await openai_api.ask(session, messages, model="gpt-4.1-nano", timeout=30)
            return result


        agent = await chat_agent_executor.create(
            llm=llm_function,
            tools=tools,
        )


        graph = MessageGraph()
        graph.add_node("agent", agent)
        graph.set_entry_point("agent")

        app = graph.compile(checkpointer=NoOpCheckpoint())
        print("🔄 Ready to chat, send message!:")

        while True:
            user_input = input("👤 You: ")
            if user_input.lower() in ("exit", "quit"):
                break

            result = await app.ainvoke(
                input=[{"role": "user", "content": user_input}]
            )
            reply = result[-1]['content']
            print("🤖 Bot:", reply)


if __name__ == "__main__":
    asyncio.run(run_chat())
