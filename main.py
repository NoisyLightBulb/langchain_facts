from langchain.document_loaders import TextLoader
from dotenv import load_dotenv

#load environment variables
load_dotenv()

#read text file
loader = TextLoader("facts.txt")
docs = loader.load()

print(docs)
