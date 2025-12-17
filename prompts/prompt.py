from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages=[
    SystemMessage(content='You are a helpful assistant'),
]

while True:
    user_input:str = input("You: ")
    if user_input == "exit":
        break
    messages.append(HumanMessage(content=user_input))
    result = model.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print("AI:  ",result.content)

print("Messages: ",messages)