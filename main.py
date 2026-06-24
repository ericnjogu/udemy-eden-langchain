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


def retrieval_chain_without_lce(query:str):
    """simple retrieval chain without without langchain expression language
    Manually retrieves docs, formats, generates a response
    """
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content


if __name__ == '__main__':
    query = "Do you server wine?"
    print('retrieving...')
    print("\n" + '=' * 70)
    result_raw = retrieval_chain_without_lce(query)
    print("\nAnswer:")
    print(result_raw)