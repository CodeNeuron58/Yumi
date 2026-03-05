from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv
load_dotenv()


with open("prompts/personality.txt") as f:
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