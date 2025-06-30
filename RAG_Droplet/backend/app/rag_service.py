import os
import pickle
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from .config import settings

class RAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY, model = "text-embedding-3-small")
        self.llm = OpenAI(temperature=0.7, api_key=settings.OPENAI_API_KEY, model="gpt-4o-mini")
        self.vector_stores = {}  # Store vector stores by session_id
        print(f"openai key:{settings.OPENAI_API_KEY}")
   
    def create_vector_store(self, documents: List[Document], session_id: str):
        """Create FAISS vector store for a session"""
        if not documents:
            raise ValueError("No documents provided")
            
        # Create FAISS vector store
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # Store in memory for quick access
        self.vector_stores[session_id] = vector_store
        
        # Save to disk for persistence
        persist_directory = f"{settings.VECOTR_PERSIST_DIR}/{session_id}"
        os.makedirs(persist_directory, exist_ok=True)
        
        # Save FAISS index
        vector_store.save_local(persist_directory)
        
        print(f"✅ FAISS vector store created with {len(documents)} documents")
        return vector_store
    
    def get_vector_store(self, session_id: str):
        """Get FAISS vector store for a session"""
        if session_id in self.vector_stores:
            return self.vector_stores[session_id]
        
        # Try to load from disk
        persist_directory = f"{settings.VECOTR_PERSIST_DIR}/{session_id}"
        if os.path.exists(persist_directory):
            try:
                vector_store = FAISS.load_local(
                    persist_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.vector_stores[session_id] = vector_store
                print(f"✅ Loaded FAISS vector store from disk for session {session_id}")
                return vector_store
            except Exception as e:
                print(f"⚠️ Failed to load FAISS store: {e}")
                return None
        
        return None
    
    def query(self, question: str, session_id: str) -> Dict[str, Any]:
        """Query the FAISS RAG system"""
        vector_store = self.get_vector_store(session_id)
        
        if not vector_store:
            return {
                "response": "No documents uploaded yet. Please upload PDFs first.",
                "sources": []
            }
        
        try:
            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            # Get response
            result = qa_chain.invoke({"query": question})
            
            # Extract sources
            sources = []
            for doc in result.get("source_documents", []):
                source_info = doc.metadata.get("source", "Unknown")
                if source_info not in sources:
                    sources.append(source_info)
            
            return {
                "response": result["result"],
                "sources": sources
            }
            
        except Exception as e:
            print(f"❌ Error in FAISS query: {e}")
            return {
                "response": f"Error processing query: {str(e)}",
                "sources": []
            }
    
    def similarity_search(self, query: str, session_id: str, k: int = 3):
        """Direct similarity search (useful for debugging)"""
        vector_store = self.get_vector_store(session_id)
        if not vector_store:
            return []
        
        try:
            docs = vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            return []