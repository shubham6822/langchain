from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7,max_output_tokens=10)

class FeedBack(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

str_parser = StrOutputParser()
py_parser = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
    template="Give me the sentiment of the following feedback \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": py_parser.get_format_instructions()},
)

positive_prompt = PromptTemplate(
    template="Write an appropriate response to this positive feedback : \n {feedback}",
    input_variables=["feedback"], 
)

negative_prompt = PromptTemplate(
    template="Write an appropriate response to this negative feedback: \n {feedback}",
    input_variables=["feedback"],   
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', positive_prompt | model | str_parser),
    (lambda x: x.sentiment == 'negative', negative_prompt | model | str_parser),
    lambda x: "No valid sentiment found."
)

chain_classifier = prompt1 | model | py_parser

main_chain = chain_classifier | branch_chain

print(main_chain.invoke({"feedback": "I love the new features of your product!"}))

