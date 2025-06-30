import os
import uuid
import shutil
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import ChatMessage, ChatResponse, UploadResponse
from .rag_service import RAGService
from .pdf_processor import PDFProcessor
from .config import settings

app = FastAPI(title="RAG Chatbot API")

# CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
try:
    from .rag_service import RAGService
    rag_service = RAGService()
    print("✅ FAISS RAG service initialized successfully")
except Exception as e:
    print(f"❌ Error initializing FAISS RAG service: {e}")
    rag_service = None

try:
    pdf_processor = PDFProcessor()
    print("✅ PDF processor initialized successfully")
except Exception as e:
    print(f"❌ Error initializing PDF processor: {e}")
    pdf_processor = None

# Create upload directory
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECOTR_PERSIST_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running", "status": "healthy"}

@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process PDF files"""
    print(f"📤 Upload request received with {len(files)} files")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if not rag_service or not pdf_processor:
        raise HTTPException(status_code=500, detail="Services not initialized properly")
    
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(settings.UPLOAD_DIR, session_id)
    
    try:
        os.makedirs(session_dir, exist_ok=True)
        print(f"📁 Created session directory: {session_dir}")
        
        processed_files = []
        saved_paths = []
        
        # Save uploaded files
        for file in files:
            print(f"📄 Processing file: {file.filename}")
            
            if not file.filename:
                continue
                
            if not file.filename.lower().endswith('.pdf'):
                print(f"⚠️ Skipping non-PDF file: {file.filename}")
                continue
            
            # Read file content
            try:
                content = await file.read()
                if len(content) > settings.MAX_FILE_SIZE:
                    print(f"⚠️ File too large: {file.filename}")
                    continue
                
                file_path = os.path.join(session_dir, file.filename)
                
                with open(file_path, "wb") as buffer:
                    buffer.write(content)
                
                saved_paths.append(file_path)
                processed_files.append(file.filename)
                print(f"✅ Saved file: {file_path}")
                
            except Exception as e:
                print(f"❌ Error processing file {file.filename}: {e}")
                continue
        
        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid PDF files found")
        
        print(f"📚 Processing {len(saved_paths)} PDFs...")
        
        # Process PDFs and create vector store
        try:
            documents = pdf_processor.process_pdfs(saved_paths)
            print(f"📄 Created {len(documents)} document chunks")
            
            rag_service.create_vector_store(documents, session_id)
            print(f"🔍 Vector store created for session: {session_id}")
            
        except Exception as e:
            print(f"❌ Error processing PDFs: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")
        
        response = UploadResponse(
            message=f"Successfully processed {len(processed_files)} files",
            files_processed=processed_files,
            session_id=session_id
        )
        
        print(f"✅ Upload completed successfully: {response}")
        return response
        
    except HTTPException:
        # Clean up on error
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        raise
    except Exception as e:
        # Clean up on error
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Chat with the RAG system"""
    print(f"💬 Chat request: {message.message[:50]}...")
    
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        result = rag_service.query(message.message, message.session_id)
        print(f"🤖 Generated response with {len(result.get('sources', []))} sources")
        
        return ChatResponse(
            response=result["response"],
            sources=result["sources"]
        )
        
    except Exception as e:
        print(f"❌ Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    status = {
        "status": "healthy",
        "rag_service": rag_service is not None,
        "pdf_processor": pdf_processor is not None,
        "upload_dir": os.path.exists(settings.UPLOAD_DIR),
        "faiss_dir": os.path.exists(settings.VECOTR_PERSIST_DIR)
    }
    print(f"🏥 Health check: {status}")
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)