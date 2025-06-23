import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# Step 1: Load PDF Files from a Directory
pdf_dir = "pdfs"  # change to your folder name
all_docs = []

for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)

# Step 2: Convert Documents into Chunks
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
chunks = text_splitter.split_documents(all_docs)
print(chunks)

# Step 3: Create Embeddings and Store in Vector Database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)

# Step 4: Set Up the Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
retriever = vector_db.as_retriever()

# Step 5: Create RAG Chain
prompt = PromptTemplate.from_template(
    "You are a helpful AI assistant. Based on the following retrieved context, answer the question concisely.\n\n"
    "Context:\n{context}\n\n"
    "Question: {input}\n"
    "Answer:"
)
stuff_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, stuff_chain)

# Step 6: Ask Multiple Questions in a Loop
print("You can now ask questions about the PDF documents. Type 'exit' to quit.\n")

while True:
    query = input("Enter your question: ")
    if query.strip().lower() in ("exit", "quit"):
        print("Exiting...")
        break

    response = rag_chain.invoke({"input": query})
    print("\nAnswer:", response["answer"])
    print("-" * 50)