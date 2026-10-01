import ollama

MODEL = "qwen3:8b"

def echo_fun(content: str):
    return content

TOOLS = [echo_fun]

SYSTEM_PROMPT = """
You are an echo agent.
Always use echo_fun to echo the user's message.
"""

def agent_run():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("Local Qwen Echo Agent")
    print("Type 'exit' to quit.")

    while True:
        prompt = input("Message: ").strip()

        if prompt.lower() == "exit":
            break

        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS
        )

        messages.append(response["message"])

        # Execute tool
        for call in response["message"].get("tool_calls", []):
            args = call["function"]["arguments"]

            result = echo_fun(**args)

            print("Agent:", result)

if __name__ == "__main__":
    agent_run()