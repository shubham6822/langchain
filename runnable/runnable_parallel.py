from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template1 = PromptTemplate(
    template="Generate a tweet related to this topic: {topic}",
    input_variables=["topic"],
)

template2  = PromptTemplate(
    template="genrate a LinkedIn post related to this topic: {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

parallel_runnable = RunnableParallel({
    "tweet":RunnableSequence(template1,model,parser),
    "linkedin_post":RunnableSequence(template2,model,parser),
})

print(parallel_runnable.invoke({"topic": "Artificial Intelligence"}))