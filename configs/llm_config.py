import os
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
openai_llm = ChatOpenAI(
    openai_api_key= OPENAI_API_KEY,
    temperature=0.4,
    model_name="gpt-4o",  
)
groq_llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    temperature=0.5,
    model_name="qwen-2.5-32b",  
)


