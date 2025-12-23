from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema


load_dotenv()


llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it",
    task="text-generation",
    
)


model = ChatHuggingFace(llm=llm)



schemas = [
    ResponseSchema(name="fact_1",description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2",description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3",description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)

template = PromptTemplate(
    template="Provide three interesting facts about the topic: {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


chain = template | model | parser

result = chain.invoke({"topic": "Space Exploration"})

print(result)