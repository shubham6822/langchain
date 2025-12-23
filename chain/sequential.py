from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7,max_output_tokens=10)

template1 = PromptTemplate(
    template="Write a detail report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Summarize the following text in one sentence: {text}",
    input_variables=["text"]
)
    
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "artificial intelligence"})
print(result)
