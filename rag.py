# Imports core LlamaIndex classes used for document loading, indexing, and prompting (a framework for LLM-based applications)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

# Loads all files from the 'docs' directory into memory as document objects
documents = SimpleDirectoryReader('docs').load_data()
# Converts the documents into vector embeddings and stores them in a searchable index
index = VectorStoreIndex.from_documents(documents)

# Template to format prompts and user input.
# Needed to override the model's default prompt, which was causing it
#to ignore the docs and answer based on its own prior training/prior "knowledge."
chat_prompt = ChatPromptTemplate(message_templates=[
    
    # System message sets strict behavior rules for the LLM
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="You are a Q&A assistant. Answer ONLY using the context provided. If the answer is not in the context, say 'I don't know based on the provided documents.' Do not use any prior knowledge."
    ),

    # User message defines how retrieved document context and user questions
    # are inserted into the final prompt sent to the model
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

# Creates a query engine used to search the vector index and generate responses
query_engine = index.as_query_engine()
# Replaces the default response prompt with the custom RAG-focused prompt above
query_engine.update_prompts({"response_synthesizer:text_qa_template": chat_prompt})

# For debugging to inspect the active prompts being used
# print(query_engine.get_prompts())

# Example of a single query to test the system (uncomment to run)
# response = query_engine.query('What is the capital of Germany?')
# print(response)

# Intro message when program starts
print("RAG Demo - Ask a question about the documents in the 'docs' directory. Type 'quit' to exit.")

# Interactive terminal loop (terminal-based Q&A)
while True:
    # capture user input
    question = input("You: ")
    # exit condition for the loop (user types 'quit')
    if question.lower() == "quit":
        break
    # search indexed docs, generates a response, prints
    response = query_engine.query(question)
    print(f"\nAssistant: {response}\n")