from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnablePassthrough,RunnableParallel,RunnableBranch
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = PromptTemplate(
    template="write a details report on {topic}",
    input_variables=["topic"],
)

template1 = PromptTemplate(
    template="summarize the following text in less than 50 words: {text}",
    input_variables=["text"],
)

template2 = PromptTemplate(
    template="summarize the following text in more than 100 words: {text}",
    input_variables=["text"],
)   

parser = StrOutputParser()

report_chain=RunnableSequence(template,model,parser)

conditional_chain = RunnableBranch(
    (lambda x: len(str(x).split(" ")) > 100,RunnableSequence(template2,model,parser)),
    (lambda x : len(str(x).split(" ")) <= 100,RunnableSequence(template1,model,parser)),
    RunnableLambda(lambda x: "Input text is empty")
)

main_chain=RunnableSequence(
    report_chain,
    conditional_chain
)

result=main_chain.invoke("Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. These intelligent machines can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI can be categorized into two main types: narrow AI, which is designed for specific tasks, and general AI, which has the ability to perform any intellectual task that a human can do. The development of AI has led to significant advancements in various fields, including healthcare, finance, transportation, and entertainment. However, it also raises ethical concerns regarding privacy, job displacement, and the potential for misuse. As AI continues to evolve, it is crucial to ensure that its development is guided by ethical principles and that its benefits are accessible to all.")
print(result)

