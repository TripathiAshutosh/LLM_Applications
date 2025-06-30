import os
import uuid
from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def process_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """Process multiple PDFs and return documents"""
        documents = []
        
        for pdf_path in pdf_paths:
            try:
                text = self.extract_text_from_pdf(pdf_path)
                filename = os.path.basename(pdf_path)
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                # Create documents with metadata
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
                    
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
                
        return documents