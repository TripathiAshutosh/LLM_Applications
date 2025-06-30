import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';
import MessageList from './components/MessageList';
import { uploadFiles, sendMessage } from './services/api';
import './App.css';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' or 'chat'

  const handleFilesUploaded = async (files) => {
    setIsUploading(true);
    try {
      const result = await uploadFiles(files);
      setSessionId(result.session_id);
      setUploadedFiles(result.files_processed);
      
      setMessages([{
        type: 'system',
        text: `Successfully uploaded and processed ${result.files_processed.length} files: ${result.files_processed.join(', ')}`,
        timestamp: new Date(),
      }]);
      
      // Auto-switch to chat tab after successful upload
      setActiveTab('chat');
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        text: `Error uploading files: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date(),
      }]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSendMessage = async (message) => {
    setMessages(prev => [...prev, {
      type: 'user',
      text: message,
      timestamp: new Date(),
    }]);

    setIsLoading(true);
    try {
      const result = await sendMessage(message, sessionId);
      setMessages(prev => [...prev, {
        type: 'assistant',
        text: result.response,
        sources: result.sources,
        timestamp: new Date(),
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        text: `Error: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date(),
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewSession = () => {
    setSessionId(null);
    setMessages([]);
    setUploadedFiles([]);
    setActiveTab('upload');
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo">🤖</div>
            <div>
              <h1>RAG Chatbot</h1>
              <p>Upload documents and chat with AI</p>
            </div>
          </div>
          
          {sessionId && (
            <button onClick={handleNewSession} className="new-session-btn">
              <span>🔄</span> New Session
            </button>
          )}
        </div>
      </header>

      <main className="app-main">
        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button 
            className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
            disabled={isUploading}
          >
            <span>📁</span> Upload Documents
          </button>
          <button 
            className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
            disabled={!sessionId || isUploading}
          >
            <span>💬</span> Chat
          </button>
        </div>

        {/* Content Area */}
        <div className="content-area">
          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div className="upload-section">
              <div className="section-header">
                <h2>📄 Upload Your Documents</h2>
                <p>Upload PDF files to create your knowledge base</p>
              </div>
              
              <FileUpload 
                onFilesUploaded={handleFilesUploaded}
                isUploading={isUploading}
              />
              
              {uploadedFiles.length > 0 && (
                <div className="uploaded-files">
                  <h3>✅ Uploaded Files ({uploadedFiles.length})</h3>
                  <div className="file-list">
                    {uploadedFiles.map((file, index) => (
                      <div key={index} className="file-item">
                        <span className="file-icon">📄</span>
                        <span className="file-name">{file}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div className="chat-section">
              <div className="section-header">
                <h2>💬 Chat with Your Documents</h2>
                <p>Ask questions about your uploaded documents</p>
              </div>
              
              {sessionId ? (
                <>
                  <div className="chat-container">
                    <div className="messages-container">
                      <MessageList messages={messages} />
                    </div>
                    
                    <ChatInterface 
                      sessionId={sessionId}
                      onSendMessage={handleSendMessage}
                      isLoading={isLoading}
                    />
                  </div>
                </>
              ) : (
                <div className="no-session">
                  <div className="no-session-content">
                    <span className="no-session-icon">📁</span>
                    <h3>No Documents Uploaded</h3>
                    <p>Please upload some PDF documents first to start chatting</p>
                    <button 
                      onClick={() => setActiveTab('upload')}
                      className="upload-redirect-btn"
                    >
                      Go to Upload
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Status Bar */}
        {sessionId && (
          <div className="status-bar">
            <div className="status-item">
              <span className="status-label">Session:</span>
              <span className="status-value">{sessionId.substring(0, 8)}...</span>
            </div>
            <div className="status-item">
              <span className="status-label">Files:</span>
              <span className="status-value">{uploadedFiles.length}</span>
            </div>
            <div className="status-item">
              <span className="status-label">Messages:</span>
              <span className="status-value">{messages.length}</span>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;