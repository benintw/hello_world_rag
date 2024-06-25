"""
This works.

"""

# Imports

from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# dont use this embeddings
# from langchain.embeddings import GPT4AllEmbeddings
# from langchain_community.embeddings import GPT4AllEmbeddings

# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA


# Instantiate the Ollama model
ollama = Ollama(
    # base_url="http://localhost:3000",
    model="llama3"
)

# Load the webpage content using WebBaseLoader
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Tsai_Ing-wen")
data = loader.load()

# Split the text into manageable chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Instantiate SentenceTransformers embeddings with the specified model
sentence_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the vector store from the document splits and embeddings
vectorstore = Chroma.from_documents(documents=all_splits, embedding=sentence_embeddings)

# Create a QA chain using the Ollama model and the vector store retriever
qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())


# question = "When was Tsai born?"
question = "What is Tsai's energy policy?"
print(qachain({"query": question}))
