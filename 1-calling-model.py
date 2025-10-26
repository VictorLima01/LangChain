import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

# Carrega as variáveis do arquivo .env
load_dotenv()

# Verifica se a chave foi carregada
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ ERRO: A variável OPENAI_API_KEY não foi encontrada no ambiente (.env).")

# Inicializa o modelo GPT
llm = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.7,
)

# Cria o prompt
template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# Formata o texto
text = template.format(product="colorful socks")

# Invoca o modelo
result = llm.invoke(text)

# Exibe o resultado
print("=== LLM RESULT ===")
print(f"Content: {result.content}\n")

if getattr(result, "usage_metadata", None):
    print("=== USAGE METADATA ===")
    for k, v in result.usage_metadata.items():
        print(f"{k}: {v}")
    print()
