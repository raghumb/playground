from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.chat_models import ChatOCIGenAI
from langchain_chroma.vectorstores import Chroma
import chromadb

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSerializable
)
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.output_parsers import XMLOutputParser
from langchain.schema import Document

import chromadb
import os
from typing import List
from typing import Optional
from typing import Dict
import asyncio
from operator import itemgetter
from langchain_community.vectorstores.pgvector import BaseModel
from langchain_core.pydantic_v1 import BaseModel, Field

from ingestion import ingest_data

def initialize_embedding_model():
        embedding_model = OCIGenAIEmbeddings(
            model_id="cohere.embed-english-light-v3.0",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id="<COMPARTMENT_ID>",
            auth_profile="<YOUR_PROFILE>" 
        )
        return embedding_model


def initialize_query_model():
    query_model = ChatOCIGenAI(
            model_id="meta.llama-3-70b-instruct",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id="<COMPARTMENT_ID>",
            model_kwargs={"temperature": 1, "max_tokens": 500},
            auth_profile="<YOUR_PROFILE>" 
        )
    return query_model

def initialize_vector_store(embedding_model, collection_name):
        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_or_create_collection(collection_name)
        vector_store = Chroma(
                client = persistent_client,
                collection_name = collection_name,
                embedding_function = embedding_model,
                collection_metadata = {"hnsw:space": "cosine", "hnsw:construction_ef": 50, "hnsw:M": 32, "search_ef": 15}, 
        )
        return vector_store

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_docs_xml(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        doc_str = f"""\
            <source id=\"{i}\">
                <article_snippet>{doc.page_content}</article_snippet>
            </source>"""
        formatted.append(doc_str)
    return "\n\n<sources>" + "\n".join(formatted) + "</sources>"
    

def predict_with_citation(query: str, vector_store, query_model, embedding_model):
        

        template = """
            You are an assistant who is expert in answering questions based on given context.
            Use the provided context only to answer the following question. Say 'I dont know, if you are unable to find the relevant document'.
            You must return both an answer and citations. A citation consists of a VERBATIM quote that \
            justifies the answer and the ID of the quote article. Return a citation for every quote across all articles \
            that justify the answer. Use the following format for your final output:

            <cited_answer>
                <answer></answer>
                <citations>
                    <citation><source_id></source_id><quote></quote></citation>
                    <citation><source_id></source_id><quote></quote></citation>
                    ...
                </citations>
            </cited_answer>
            Here is the contextual information to be used: {context}

            Question: {question}
            """
        prompt = ChatPromptTemplate.from_template(template)

        embedding_vector = embedding_model.embed_query(query)
        docs = vector_store.similarity_search_by_vector(embedding = [embedding_vector], k = 5)
        print("Number of documents returned " + str(len(docs)))
        chain = (
            RunnablePassthrough.assign(context=lambda input: format_docs_xml(input["context"]))
            | prompt
            | query_model
            | XMLOutputParser()
        )
        response = chain.invoke({"context": docs, "question": query})
        print(response)
        return response

embedding_model = initialize_embedding_model()
query_model = initialize_query_model()
vector_store = initialize_vector_store( embedding_model= embedding_model, collection_name = "my_collection_2")

ingest_data(vector_store)
query = "What are components of Open Telemetry?"
predict_with_citation(query, vector_store, query_model, embedding_model)