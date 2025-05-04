from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below:

Here is the conversation history: {context}

Prompt: {prompt}

Answer:
"""

model = OllamaLLM(model="local_llm")

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome to Machine Learning tasks assistance! Type 'exit' to end the session.")
    while True:
        user_input = input("Engineer: ")
        if user_input.lower() == "exit":
            break
        result = chain.invoke({"context": context, "prompt": user_input})
        print("ML Assistant: ", result)
        context += f"\nEngineer: {user_input}\nML Assistant: {result}"

if __name__ == "__main__":
    handle_conversation()