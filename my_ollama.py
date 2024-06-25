"""
This works.

"""

from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.embeddings import GPT4AllEmbeddings
# from langchain_community.embeddings import GPT4AllEmbeddings

# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA


ollama = Ollama(
    # base_url="http://localhost:3000",
    model="llama3"
)

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Tsai_Ing-wen")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

sentence_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore = Chroma.from_documents(documents=all_splits, embedding=sentence_embeddings)

qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())


# print(ollama.invoke("Why is the sky blue?"))

# question = "When was Tsai born?"
question = "What is Tsai's energy policy?"
print(qachain({"query": question}))
