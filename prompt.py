from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv

#load environment variables
load_dotenv()

#initialize chat model
chat = ChatOpenAI()

#initialize embeddings
embeddings = OpenAIEmbeddings()

#load chroma databse from emb folder
db = Chroma(
    persist_directory = "emb",
    embedding_function = embeddings
)

#initializing retriever
retriever = RedundantFilterRetriever(
    embeddings = embeddings,
    chroma = db
    )

#creating chain
chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever,
    chain_type = "stuff"
)


result = chain.run("What is an interesting fact about the English language?")

print(result)
