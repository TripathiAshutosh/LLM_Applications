import React, { useState } from 'react';

const ChatInterface = ({ sessionId, onSendMessage, isLoading }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-interface">
      <form onSubmit={handleSubmit} className="chat-form">
        <div className="input-group">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={sessionId ? "Ask a question about your documents..." : "Please upload PDFs first"}
            disabled={!sessionId || isLoading}
            className="chat-input"
          />
          <button 
            type="submit" 
            disabled={!input.trim() || !sessionId || isLoading}
            className="send-button"
          >
            {isLoading ? '...' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;