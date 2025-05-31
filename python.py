"""


# Initialize an empty chat history
chat_history = {
    "messages":[],
}

while True:
    arr = ["exit", "quit"]
    user_input = input("User: ")


    for command in arr:
        if command in user_input.lower():
            print("Exiting chat. Here's the chat history:")
            print(chat_history)
            break
    else:
        chat_history["messages"] = chat_history["messages"]+[user_input]

"""



dicti = {
    "messages": [
        {"role": "user", "content": "My name is Suraj Jena"},
    ]
}

dicti["messages"] = ["hello"]

print(dicti)

