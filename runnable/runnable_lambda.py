from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnablePassthrough,RunnableParallel
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = PromptTemplate(
    template="Make a joke of {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

joke_chain = RunnableSequence(template,model,parser)

parallel_chain= RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count":RunnableLambda(lambda x: len(x.split(" ")))
})

main_chain=RunnableSequence(
    joke_chain,
    parallel_chain
)

result=main_chain.invoke("Why did the chicken cross the road?")
print(result)