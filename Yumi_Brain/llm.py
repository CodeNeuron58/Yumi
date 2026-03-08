from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
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


class YumiResponse(BaseModel):
    response_text: str = Field(description="The conversational, concise text Yumi will speak out loud.")
    expression: str = Field(description="The generic facial expression label (e.g. smile, angry, sad).")
    motion: str = Field(description="The generic body motion label (e.g. nod, tilthead, fidget).")


# Initialize the LLM with structured output bounds
base_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# Apply Pydantic schema
llm = base_llm.with_structured_output(YumiResponse)

chain = prompt | llm