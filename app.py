import os
import time
import gradio as gr
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API Key from Hugging Face Secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found. Add it in Hugging Face Space Secrets.")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that generates comprehensive and well-structured notes on a given topic."),
    ("user", "Generate detailed notes on the following topic: {topic}")
])

# Chain
output_parser = StrOutputParser()
note_generation_chain = prompt | llm | output_parser

# Function with retry logic
def generate_notes_gradio(topic):
    if not topic.strip():
        return "⚠️ Please enter a topic."

    retries = 3

    for attempt in range(retries):
        try:
            notes = note_generation_chain.invoke({"topic": topic})
            return notes

        except Exception as e:
            if "503" in str(e) and attempt < retries - 1:
                time.sleep(2)
            else:
                return "⚠️ Server busy. Please try again."

# Gradio UI
iface = gr.Interface(
    fn=generate_notes_gradio,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter a topic (e.g., Artificial Intelligence, Python...)",
        label="📌 Topic"
    ),
    outputs=gr.Markdown(label="📝 Generated Notes"),
    title="📚 AI Notes Generator",
    description="Generate structured notes using Gemini + LangChain",
    theme="soft"
)

# Launch app
if __name__ == "__main__":
    iface.launch()
