from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

def get_llm():

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )
    return llm
    