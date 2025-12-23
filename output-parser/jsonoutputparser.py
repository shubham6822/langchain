from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()


llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it",
    task="text-generation",
    
)


model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="given a name ,age and a city of any fictional charactter \n {format_instructions}",
    input_variables=[""],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template1 | model | parser

result = parser.parse(chain.invoke({}))

print(result)