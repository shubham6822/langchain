from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7,max_output_tokens=10)

result = model.invoke("write a one funny joke on llm")

print(result.content)