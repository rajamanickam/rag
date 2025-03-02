#Sample RAG code explain https://www.blog.qualitypointtech.com/2025/03/how-to-build-simple-retrieval-augmented.html
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

#Before running this code make sure that you exported the HuggingFace API token in the environment.

# Step 1: Load and Prepare Documents
documents = [
    " blah blah blah blah blah blah. blah blah blahblah blah blah. Vitamin C helps boost immunity.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.",
    "blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah. Exercise improves mental and physical health. blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.",
    "blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah. Drinking enough water keeps you hydrated and improves focus. blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah.blah blah blah blah blah blah. blah blah blahblah blah blah."
]

# Step 2: Convert Documents into Chunks
text_splitter = CharacterTextSplitter(chunk_size=60, chunk_overlap=10, separator=".")
chunks = text_splitter.create_documents(documents)
print(chunks)

# Step 3: Create Embeddings and Store in Vector Database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)

# Step 4: Set Up the Chat Model
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-alpha")
retriever = vector_db.as_retriever(search_kwargs={"k": 1}) 

# Step 5: Create RAG Chain
prompt = PromptTemplate.from_template(
    "You are a helpful AI assistant. Based on the following retrieved context, answer the question concisely.\n\n"
    "Context:\n{context}\n\n"
    "Question: {input}\n"
    "Answer:"
)

stuff_chain = create_stuff_documents_chain(llm, prompt)



# Step 6: Debug - Print what is being sent to the model
def debug_rag_chain(input_query,retrieved_text):
    """Retrieve documents, format the prompt, and print before calling LLM"""
    
    # Retrieve relevant documents
    
    
    # Format the final prompt
    formatted_prompt = prompt.format(context=retrieved_text, input=input_query)
    
    # Print the final prompt sent to the LLM
    print("\n===== DEBUG: FINAL PROMPT SENT TO LLM =====\n")
    print(formatted_prompt)
    print("\n==========================================\n")


   
# Step 7: Run query and print response
query = "How does exercise affect health?"
retrieved_docs = retriever.get_relevant_documents(query)

# Extract retrieved text
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
debug_rag_chain(query,retrieved_text) 
rag_chain = create_retrieval_chain(retriever, stuff_chain)
response = rag_chain.invoke({"input": query})
print("Final Answer:", response["answer"])
