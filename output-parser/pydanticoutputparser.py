from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()


llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it",
    task="text-generation",
    
)


model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(gt=18, description="The person's age")
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
    )

chain = template | model | parser

output = chain.invoke({"place": "Italian"})

print(output)