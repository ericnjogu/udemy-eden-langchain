from operator import itemgetter
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage
from langchain_core.tools import retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone, PineconeVectorStore
from openai.types import embedding
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from constants import INDEX_NAME

load_dotenv()

print("initializing components")
embeddings = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=1024);
llm = ChatOpenAI()
vector_store = PineconeVectorStore(index_name=os.environ.get('INDEX_NAME', INDEX_NAME), embedding=embeddings)
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


def retrieval_chain_with_lce():
    """create a retrieval chain using LCEL
    returns a chain that can be invoked with {"question": "...."}
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
            )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == '__main__':
    query = "Do you serve wine?"
    print('retrieving...')
    print("\n" + '=' * 70)
    result_raw = retrieval_chain_with_lce().invoke({"question": query})
    print("\nAnswer:")
    print(result_raw)