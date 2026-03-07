from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

from dotenv import load_dotenv
load_dotenv()


current_dir = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(current_dir, "prompts", "personality.txt")
with open(prompt_path) as f:
    personality = f.read()

prompt = ChatPromptTemplate.from_messages([
    ("system", personality),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

chain = prompt | llm