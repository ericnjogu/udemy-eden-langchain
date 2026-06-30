import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai.types import embedding
from constants import INDEX_NAME

load_dotenv()

if __name__ == '__main__':
    print('ingesting...')
    loader = TextLoader(f"{os.environ['TXT_DIR']}/drinks.txt")
    document = loader.load()
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'], 
        model='text-embedding-3-large',dimensions=1024)

    loader = TextLoader(f"{os.environ['TXT_DIR']}/drinks.txt")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get('INDEX_NAME', INDEX_NAME))
    print('completed ingesting...')