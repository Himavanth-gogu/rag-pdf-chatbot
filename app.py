from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ctransformers import AutoModelForCausalLM

print("Reading your book...")

# Load PDF
loader = PyPDFLoader("book.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Create vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# Load TinyLlama (optimized for CPU laptops)
llm = AutoModelForCausalLM.from_pretrained(
    "models",
    model_file="tinyllama.gguf",
    model_type="llama",
    gpu_layers=0,
    threads=6,
    max_new_tokens=80,
    temperature=0.0
)

print("🤖 Real AI PDF Chatbot Ready! Ask questions (type 'exit' to stop)")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        break

    if question == "":
        print("(please type a question)")
        continue

    # Retrieve relevant context
    results = db.similarity_search(question, k=1)
    context = results[0].page_content[:800]

    # Strict anti-hallucination prompt
    prompt = f"""
You are a helpful document assistant.

Answer ONLY using the given context.
Do not use outside knowledge.
If answer is not clearly present, reply exactly: Not found in document.
Keep answer short and clear (2 sentences max).

Context:
{context}

Question: {question}

Answer:
"""

    print("Bot is thinking... (wait few seconds)")
    raw_output = llm(prompt)

    # Clean response
    answer = raw_output.strip()

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    if answer == "":
        answer = "Not found in document"

    print("Bot:", answer)
