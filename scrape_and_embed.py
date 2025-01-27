# scrape_and_embed.py
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def scrape_data(url):
    # Initialize the URL loader
    loader = WebBaseLoader(url)
    
    # Extract documents from the URL
    documents = loader.load()
    return documents

def create_embeddings(documents):
    # Use model name as a keyword argument
    model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    
    # Pass the model name to HuggingFaceEmbeddings as a keyword argument
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    
    # Store the embeddings in a FAISS vector store
    vectorstore = FAISS.from_documents(documents, embedding)
    
    # Save the vector store locally
    vectorstore.save_local("vectorstore")
    
if __name__ == "__main__":
    url = "https://brainlox.com/courses/category/technical"
    
    # Scrape the data
    documents = scrape_data(url)
    
    # Create and store embeddings
    create_embeddings(documents)
    
    print("Data scraped and embeddings created successfully!")
