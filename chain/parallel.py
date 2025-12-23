from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7,max_output_tokens=10)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="create a quiz with 3 questions from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="merge the provide note and quiz into a single study guide \n Notes: {notes} \n Quiz: {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parellel_chain = RunnableParallel({
    "notes": prompt1 | model | parser,
    "quiz": prompt2 | model | parser
})

merger_chain = prompt3 | model | parser

main_chain = parellel_chain | merger_chain

result = main_chain.invoke({"text": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. These intelligent machines can perform tasks such as problem-solving, decision-making, language understanding, and visual perception. AI technologies include machine learning, natural language processing, robotics, and computer vision. The goal of AI is to create systems that can operate autonomously and improve their performance over time through experience."})

print(result)

# main_chain.get_graph().print_ascii()    

