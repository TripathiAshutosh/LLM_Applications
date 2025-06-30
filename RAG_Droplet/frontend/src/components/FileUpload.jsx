import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const FileUpload = ({ onFilesUploaded, isUploading }) => {
  const onDrop = useCallback((acceptedFiles) => {
    const pdfFiles = acceptedFiles.filter(file => 
      file.type === 'application/pdf'
    );
    
    if (pdfFiles.length > 0) {
      onFilesUploaded(pdfFiles);
    }
  }, [onFilesUploaded]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: true,
    disabled: isUploading
  });

  return (
    <div className="file-upload-container">
      <div 
        {...getRootProps()} 
        className={`dropzone ${isDragActive ? 'active' : ''} ${isUploading ? 'disabled' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-content">
          <div className="upload-icon">📄</div>
          {isUploading ? (
            <p>Uploading and processing files...</p>
          ) : isDragActive ? (
            <p>Drop the PDF files here...</p>
          ) : (
            <div>
              <p>Drag & drop PDF files here, or click to select</p>
              <p className="upload-hint">Support for multiple PDF files</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUpload;