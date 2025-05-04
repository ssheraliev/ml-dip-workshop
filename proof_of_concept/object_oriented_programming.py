from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below:

Here is the conversation history: {context}

Question: {prompt}

Answer:
"""

model = OllamaLLM(model="local_llm")

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversartion():
    context = ""
    print("Welcome to Python Assistant! Type 'exit' to quit.")
    while True:
        user_input = input("Engineer: ")
        if user_input.lower() == "exit":
            break
        result = chain.invoke({"context": context, "prompt": user_input})
        print("Python Assistant: ", result)
        context += f"\nUser: {user_input}\nPython Assistant: {result}"     

if __name__ == "__main__":
    handle_conversartion()  