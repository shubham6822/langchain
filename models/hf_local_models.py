from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=100
    )
)


model = ChatHuggingFace(llm=llm)


result = model.invoke("write a one funny joke on llm")


print(result.content)