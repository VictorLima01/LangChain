from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.7,
)

template = PromptTemplate(
    input_variables=["name", "question"],
    template="Hi, i'm {name}! Answer this question: {question}"
)

chain = template | llm

result = chain.invoke({"name": "Letícia", "question": "What is a good name for a company that makes colorful socks?"})

print(f"Content: {result.content}\n")