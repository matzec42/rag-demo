from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

documents = SimpleDirectoryReader('docs').load_data()
index = VectorStoreIndex.from_documents(documents)

chat_prompt = ChatPromptTemplate(message_templates=[
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="You are a Q&A assistant. Answer ONLY using the context provided. If the answer is not in the context, say 'I don't know based on the provided documents.' Do not use any prior knowledge."
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "Context:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Question: {query_str}\n"
            "Answer: "
        )
    )
])
query_engine = index.as_query_engine()
query_engine.update_prompts({"response_synthesizer:text_qa_template": chat_prompt})

# print(query_engine.get_prompts())

# response = query_engine.query('What is the capital of Germany?')
# print(response)

print("RAG Demo - Ask a question about the documents in the 'docs' directory. Type 'quit' to exit.")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    response = query_engine.query(question)
    print(f"\nAssistant: {response}\n")