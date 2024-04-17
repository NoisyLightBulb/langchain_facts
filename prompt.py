from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

#load environment variables
load_dotenv()

#initialize embeddings
embeddings = OpenAIEmbeddings()

#load chroma databse from emb folder
db = Chroma(
    persist_directory = "emb",
    embedding_function = embeddings
)

#initializing retriever
retriever = db.as_retriever()

#creating chain
chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever,
    chain_type = "stuff"
)
