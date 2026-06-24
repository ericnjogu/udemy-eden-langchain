import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage
from langchain_core.tools import retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone, PineconeVectorStore
from openai.types import embedding

load_dotenv()

print("initializing components")
embeddings = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=1024);
llm = ChatOpenAI()
vector_store = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}

    Question: {question}

    Provide a detailed answer:
    """
)

def format_docs(docs):
    """format retrieved documents into a single search string"""
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    query = "Do you server wine?"
    print('retrieving...')
    print("\n" + '=' * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)
    """
    Answer:
    I do not serve wine as I am a virtual assistant and not a physical establishment. However, I can provide you with 
    information about different types of wine and 
    recommendations for pairing wine with meals. Let me know how I can assist you further with wine-related queries.
    """