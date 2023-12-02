"""
Functions copied from GCP examples:
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/examples/langchain-intro/intro_langchain_palm_api.ipynb
"""

import time
from typing import List

from chat.models import Chat, Document, DocumentChunk, Message, User
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pgvector.django import CosineDistance
from pydantic import BaseModel

# Embedding
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
gcp_embeddings = OpenAIEmbeddings()

text_llm = OpenAI()
summarize_llm = OpenAI(model_name='gpt-3.5-turbo-16k')
summarize_chain = load_summarize_chain(summarize_llm, chain_type="map_reduce")
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def get_docs_chunks_by_embedding(request, query, max_distance=None):
    query_embedding = gcp_embeddings.embed_documents([query])[0]
    user_docs = Document.objects.filter(user=request.user)
    # documents_by_mean = user_docs.order_by(
    #     CosineDistance("mean_embedding", query_embedding)
    # )[:3]
    if max_distance is None:
        documents_by_summary = user_docs.order_by(
            CosineDistance("summary_embedding", query_embedding)
        )[:3]
        chunks_by_embedding = (
            DocumentChunk.objects.filter(document__in=user_docs)
            .order_by(CosineDistance("embedding", query_embedding))[:10]
            .prefetch_related("document")
        )
    else:
        documents_by_summary = user_docs.alias(
            distance=CosineDistance("summary_embedding", query_embedding)
        ).filter(distance__lt=max_distance)[:3]
        chunks_by_embedding = (
            DocumentChunk.objects.filter(document__in=user_docs)
            .alias(distance=CosineDistance("embedding", query_embedding))
            .filter(distance__lt=max_distance)
            .prefetch_related("document")
        )[:10]

    return documents_by_summary, chunks_by_embedding


def get_qa_response(query, documents, return_sources=True):
    if return_sources:
        chain = load_qa_with_sources_chain(text_llm, chain_type="stuff")
        response = chain(
            {"input_documents": documents, "question": query}, return_only_outputs=True
        )
        print(response)
        return response["output_text"]
    else:
        chain = load_qa_chain(text_llm, chain_type="stuff")
        response = chain.run(input_documents=documents, question=query)
        return response
