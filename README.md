# -------------------------------------------------------
# Project: Document Knowledgebase Chatbot (RAG)
# Author: Himavanth Reddy
# Description:
# This program reads a PDF file and answers user questions
# using semantic search + local language model.
# -------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ctransformers import AutoModelForCausalLM


print("\nLoading document... please wait\n")

# -------- Step 1: Read PDF --------
pdf_loader = PyPDFLoader("book.pdf")
pages = pdf_loader.load()


# -------- Step 2: Break document into small pieces --------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

sections = text_splitter.split_documents(pages)


# -------- Step 3: Convert text into vectors --------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(sections, embedding_model)


# -------- Step 4: Load local AI model --------
print("Starting local AI model...")

chat_model = AutoModelForCausalLM.from_pretrained(
    "models",
    model_file="tinyllama.gguf",
    model_type="llama",
    gpu_layers=0,
    threads=6,
    max_new_tokens=80,
    temperature=0.0
)

print("\nChatbot ready! Ask anything from the document.")
print("Type 'exit' to stop.\n")


# -------- Step 5: Chat Loop --------
while True:

    user_query = input("You: ").strip()

    if user_query.lower() == "exit":
        print("Goodbye!")
        break

    if user_query == "":
        print("Please enter a question.")
        continue

    # Find related paragraph
    search_result = vector_store.similarity_search(user_query, k=1)
    context_text = search_result[0].page_content[:800]

    # Prepare instruction
    instruction = f"""
Use only the information given below to answer.

If the answer is not present, reply:
Not found in document

Text:
{context_text}

Question:
{user_query}

Answer:
"""

    print("Thinking...\n")

    model_reply = chat_model(instruction)

    final_answer = model_reply.strip()

    if final_answer == "":
        final_answer = "Not found in document"

    print("Bot:", final_answer, "\n")

