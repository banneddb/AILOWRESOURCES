from dotenv import load_dotenv
load_dotenv()

import anthropic
client = anthropic.Anthropic()

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": user_input}],
    )

    for block in message.content:
        if block.type == "text":
            print("Claude:", block.text)
