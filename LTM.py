# retrieve memories
helpful_memories = memory.search(query=prompt, user_id="philipp")
memories_str = "\n".join(f"- {entry['memory']}" for entry in helpful_memories["results"])
# extend system prompt
extended_system_prompt = f"You are a helpful AI Assistant. You have the following memories about the user:\n{helpful_memories}"
 
# generate response
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "system_instruction": extended_system_prompt
    }
)
 
print(response.text)
# This is a great idea ... Here are a few options ... Both options allow you to subtly connect to your interests in mountain climbing