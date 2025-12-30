from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = PromptTemplate(
    template="Translate the following English text to French: {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

runnable = RunnableSequence(
    template,
    model,
    parser,
)

print(runnable.invoke({"text": "Hello, how are you?"}))