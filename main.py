from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

#load environment variables
load_dotenv()

#initialize embeddings
embeddings = OpenAIEmbeddings()

#initialize text splitter
text_splitter = CharacterTextSplitter(
    separator = "\n",                           #split on this character
    chunk_size = 200,                           #maximum chunk size
    chunk_overlap = 0
)

#read text file
loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter = text_splitter)

#initialize Chroma vector database and calculate embeddings
db = Chroma.from_documents(
    docs, #
    embedding = embeddings,
    persist_directory = "emb"
    )


results = db.similarity_search_with_score(
    "What is an interesting fact about the English language",
    k=3                                                                 #number of results
    )

# results without score
# results = db.similarity_search("What is an interesting fact about large mammals")


for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content)
