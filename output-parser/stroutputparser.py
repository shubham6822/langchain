from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it",
    task="text-generation",
    
)


model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="write a detail report about {topic}.", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="write a 5 line summary on the given text \n {text}.", input_variables=["text"]
)

parser = StrOutputParser()


chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Artificial Intelligence"})

print(result)