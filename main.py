from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OctoAIEmbeddings
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
docs = loader.load_and_split(
    text_splitter = text_splitter
    )


for doc in docs:
    print(doc.page_content)
    print("\n")
