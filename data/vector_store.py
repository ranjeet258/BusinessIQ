from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os

class VectorStoreManager:
    def __init__(self, hf_token: str):
        if not hf_token:
            raise ValueError("HuggingFace token is required to initialize vector store embeddings.")
            
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-small-en-v1.5",
            huggingfacehub_api_token=hf_token
        )
        self.vector_store: Optional[FAISS] = None
        
    def ingest_documents(self, documents: List[Document]):
        """Creates or updates the FAISS vector store with new documents."""
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
            
    def get_retriever(self, k: int = 4):
        """Returns a LangChain retriever if the vector store is initialized."""
        if self.vector_store is not None:
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        return None
