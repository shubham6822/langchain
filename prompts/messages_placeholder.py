from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import  HumanMessage, AIMessage
# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = [
    HumanMessage(content="I want to request a refund for my order #12345."),
    AIMessage(content="Your refund request for order #12345 has been initiated. It will be processed in 3-5 business days.")
]

print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

print(prompt)