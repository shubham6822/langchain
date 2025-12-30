import random

class NakliLLM:

    def __init__(self) -> None:
        print("NakliLLM initialized")

    def invoke(self) -> dict:
        response= ["This is a nakli response.", "Another nakli response.", "Yet another nakli response."]
        return {"content": random.choice(response) }


class NakliPromptTemplate:

    def __init__(self, template:str,input_variables:list) -> None:
        self.template = template
        self.input_variables = input_variables

    def format(self, input) -> str:
        return self.template.format(**input)
    

class NakliLLMChain:

    def __init__(self, llm:NakliLLM, prompt:NakliPromptTemplate) -> None:
        self.llm = llm
        self.prompt = prompt
    
    def run(self, input:dict) -> dict:
        formatted_prompt = self.prompt.format(input)
        print("Formatted Prompt: ", formatted_prompt)
        return self.llm.invoke()["content"]
    

if __name__ == "__main__":
    # llm = NakliLLM()
    # prompt = NakliPromptTemplate(
    #     template="Translate the following English text to French: {text}",
    #     input_variables=["text"]
    # )

    # user_input = "Hello, how are you?"
    # formatted_prompt = prompt.format({"text": user_input})
    # print("Formatted Prompt: ", formatted_prompt)

    # response = llm.invoke()
    # print("LLM Response: ", response["content"])

    llm = NakliLLM()
    prompt = NakliPromptTemplate(
        template="Translate the following English text to French: {text}",
        input_variables=["text"]
    )
    chain = NakliLLMChain(llm=llm, prompt=prompt)
    user_input = "Hello, how are you?"
    response = chain.run({"text": user_input})
    print("Chain Response: ", response)