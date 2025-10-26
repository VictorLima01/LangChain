import random
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
import random

load_dotenv()

total_prs = 100

@tool
def get_language_description(language: str) -> str:
    """Get the description of a programming language."""
    if language == "Python":
        return "Python is a high-level, interpreted programming language known for its readability and versatility."
    elif language == "JavaScript":
        return "JavaScript is a versatile, high-level programming language primarily used for web development to create interactive effects within web browsers."
    elif language == "Java":
        return "Java is a high-level, class-based, object-oriented programming language designed to have as few implementation dependencies as possible."

@tool
def get_pr_total_count() -> int:
    """Get the total number of pull requests in a repository."""
    global total_prs
    return int(total_prs)

@tool
def close_pr(qtd_prs: int) -> bool:
    """Close a number of pull requests."""
    global total_prs
    if random.randint(0, 1) == 0:
        return False
    total_prs -= qtd_prs
    return True

llm = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.7,
)

tools = [get_language_description, get_pr_total_count, close_pr]

prompt1 = """
You are a software engineer. Answer questions using the available tools.
Use each tool only once when needed. Give clear, direct answers.
If you have any error or false statement by a tool, you call the tool again unitl you get the correct answer.
"""

agent = create_agent(
    llm,
    tools,
    system_prompt=prompt1,
)

message = HumanMessage(content="""
    Answer: What is the description of the JavaScript programming language and how many PRs are there in the repository?
    Close 10 pull requests and tell me the new total.
""")

result = agent.invoke({"messages": [message]})

messages = result["messages"]

# Filtra apenas as mensagens do tipo AIMessage
ai_messages = [m for m in messages if m.__class__.__name__ == "AIMessage"]

# Pega a última mensagem do AI
if ai_messages:
    final_content = ai_messages[-1].content
    print("=== FINAL CONTENT ===")
    print(final_content)
else:
    print("Nenhuma mensagem de AI encontrada.")