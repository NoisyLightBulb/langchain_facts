from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
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
