

from google import genai
from mem0 import Memory
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key = os.getenv('GEMINI_API_KEY') )
config = {
    "embedder": {"provider": "gemini", "config": {"model": "models/text-embedding-004"}},
    "llm": {"provider": "gemini", "config": {"model": "gemini-2.5-flash", "temperature": 0.0, "max_tokens": 2000}},
    "vector_store": {"config": {"embedding_model_dims": 768}}
}
memory = Memory.from_config(config)
 
system_prompt = "You are a helpful AI. Answer the question based on query and memories."


def chat_with_memories(history: list[dict], user_id: str = "default_user") -> str:
    # Retrieve relevant memories
    print(history[-1]["parts"][0]["text"])
    relevant_memories = memory.search(query=history[-1]["parts"][0]["text"], user_id=user_id, limit=5)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
 
    # Generate Assistant response
    memory_system_prompt = f"{system_prompt}\nUser Memories:\n{memories_str}"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,
        config={"system_instruction": memory_system_prompt}
    )
    history.append({"role": "model", "parts": [{"text": response.text}]})
    # Create new memories from the conversation we need to convert the history to a list of messages
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": part["parts"][0]["text"]} for i, part in enumerate(history)]
    memory.add(messages, user_id=user_id)
 
    return history
 
def main():
    print("Chat with Gemini (type 'exit' to quit)")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        history.append({"role": "user", "parts": [{"text": user_input}]})
        response = chat_with_memories(history)
        print(f"Gemini: {response[-1]['parts'][0]['text']}")
 
 
main()