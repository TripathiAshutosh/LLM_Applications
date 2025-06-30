# RAG Chatbot - DigitalOcean Deployment

A full-stack RAG (Retrieval-Augmented Generation) chatbot application with FastAPI backend and React frontend, designed for easy deployment on DigitalOcean.

## Features

- **Upload Multiple PDFs**: Drag and drop interface for PDF uploads
- **RAG-based Chat**: Query your documents using OpenAI LLM and ChromaDB
- **Session Management**: Multiple document sessions with unique IDs
- **Professional UI**: Modern, responsive React interface
- **Docker Deployment**: Containerized for easy deployment
- **Production Ready**: Nginx reverse proxy with SSL support

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **LangChain**: LLM application framework
- **ChromaDB**: Vector database for embeddings
- **OpenAI**: LLM and embeddings
- **PyPDF2**: PDF text extraction

### Frontend
- **React 18**: Modern React with hooks
- **Axios**: HTTP client for API calls
- **React Dropzone**: File upload component
- **CSS3**: Modern styling with gradients and animations

### Deployment
- **Docker & Docker Compose**: Containerization
- **Nginx**: Reverse proxy and static file serving
- **DigitalOcean**: Cloud hosting platform

## Quick Start

### Prerequisites

1. **DigitalOcean Droplet** (Ubuntu 20.04+ recommended)
2. **OpenAI API Key**
3. **Domain name** (optional, for SSL)

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag-chatbot-deploy.git
cd rag-chatbot-deploy
```

2. **Set up environment variables**
```bash
# Backend
cp backend/.env.example backend/.env
# Edit backend/.env and add your OPENAI_API_KEY

# Frontend
cp frontend/.env.example frontend/.env
# Edit frontend/.env and set REACT_APP_API_URL
```

3. **Run with Docker Compose**
```bash
docker-compose up --build
```

4. Commit your changes
5. Push to your fork
6. Create a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the logs using the commands provided
3. Create an issue on GitHub with detailed information
4. Include relevant logs and error messages

## Roadmap

- [ ] Add user authentication
- [ ] Implement chat history persistence
- [ ] Add support for more document types (Word, PowerPoint)
- [ ] Implement real-time chat with WebSockets
- [ ] Add document preview functionality
- [ ] Performance monitoring and analytics
- [ ] Multi-language support
- [ ] Advanced RAG techniques (query rewriting, re-ranking)

## Acknowledgments

- OpenAI for providing the LLM API
- LangChain for the RAG framework
- ChromaDB for the vector database
- React team for the frontend framework
- FastAPI team for the backend framework **Access the application**
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## DigitalOcean Deployment

### 1. Create and Setup Droplet

```bash
# Create a new Ubuntu droplet (2GB RAM minimum recommended)
# SSH into your droplet

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again to apply docker group changes
exit
```

### 2. Deploy Application

```bash
# SSH back into your droplet
# Clone your repository
git clone https://github.com/yourusername/rag-chatbot-deploy.git
cd rag-chatbot-deploy

# Set up environment variables
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# Edit environment files
nano backend/.env
# Add: OPENAI_API_KEY=your_actual_openai_api_key

nano frontend/.env
# Add: REACT_APP_API_URL=http://YOUR_DROPLET_IP:8000

# Create a .env file for docker-compose
echo "OPENAI_API_KEY=your_actual_openai_api_key" > .env

# Build and run
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs
```

### 3. Configure Firewall

```bash
# Allow necessary ports
sudo ufw allow OpenSSH
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 8000
sudo ufw enable
```

### 4. Access Your Application

- Visit: `http://YOUR_DROPLET_IP`
- Upload PDFs and start chatting!

## Production Setup (Optional)

### SSL Certificate with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Update docker-compose.yml to use production profile
docker-compose --profile production up -d
```

### Environment Variables for Production

```bash
# backend/.env
OPENAI_API_KEY=your_openai_api_key

# frontend/.env
REACT_APP_API_URL=https://yourdomain.com/api

# .env (for docker-compose)
OPENAI_API_KEY=your_openai_api_key
```

## Project Structure

```
rag-chatbot-deploy/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI application
│   │   ├── models.py         # Pydantic models
│   │   ├── rag_service.py    # RAG implementation
│   │   ├── pdf_processor.py  # PDF handling
│   │   └── config.py         # Configuration
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile           # Backend container
│   └── .env.example         # Environment template
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── services/        # API service
│   │   ├── App.jsx          # Main App component
│   │   └── App.css          # Styles
│   ├── public/
│   ├── package.json         # Node dependencies
│   ├── Dockerfile          # Frontend container
│   └── .env.example        # Environment template
├── docker-compose.yml      # Multi-container setup
├── nginx.conf             # Reverse proxy config
└── README.md              # This file
```

## API Endpoints

### Backend API (Port 8000)

- `GET /` - Health check
- `POST /upload` - Upload PDF files
- `POST /chat` - Send chat message
- `GET /health` - Service health status
- `GET /docs` - API documentation (Swagger)

### Request/Response Examples

**Upload Files:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

**Chat:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is this document about?", "session_id": "your-session-id"}'
```

## Troubleshooting

### Common Issues

1. **Permission Denied for Docker**
```bash
sudo usermod -aG docker $USER
# Logout and login again
```

2. **Port Already in Use**
```bash
# Check what's using the port
sudo lsof -i :80
sudo lsof -i :8000

# Stop conflicting services
sudo systemctl stop apache2  # if Apache is running
sudo systemctl stop nginx    # if Nginx is running
```

3. **OpenAI API Key Issues**
```bash
# Verify your API key is set correctly
docker-compose exec backend env | grep OPENAI
```

4. **File Upload Issues**
```bash
# Check backend logs
docker-compose logs backend

# Verify upload directory permissions
docker-compose exec backend ls -la uploads/
```

5. **Frontend Not Loading**
```bash
# Check if backend is accessible
curl http://YOUR_DROPLET_IP:8000/health

# Check frontend logs
docker-compose logs frontend
```

### Logs and Monitoring

```bash
# View all service logs
docker-compose logs

# View specific service logs
docker-compose logs backend
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f backend

# Check container status
docker-compose ps
docker stats
```

### Performance Optimization

1. **Increase Droplet Resources**: 2GB+ RAM recommended for production
2. **Enable Gzip Compression**: Already configured in nginx
3. **Use CDN**: For static assets in production
4. **Database Optimization**: Consider external vector database for scale
5. **Caching**: Implement Redis for session management

## Security Considerations

1. **Environment Variables**: Never commit API keys to git
2. **CORS Configuration**: Restrict origins in production
3. **Rate Limiting**: Already configured in nginx
4. **SSL/TLS**: Use HTTPS in production
5. **Firewall**: Restrict unnecessary ports
6. **Updates**: Regularly update dependencies

## YouTube Video Outline

1. **Introduction** (2 min)
   - Project overview and demo
   - Technologies used

2. **Project Setup** (5 min)
   - File structure explanation
   - Backend development walkthrough
   - Frontend development walkthrough

3. **Local Development** (3 min)
   - Environment setup
   - Running with Docker Compose
   - Testing the application

4. **DigitalOcean Deployment** (8 min)
   - Creating droplet
   - Installing Docker
   - Deploying application
   - Configuring domain and SSL

5. **Testing and Troubleshooting** (2 min)
   - Common issues and solutions
   - Performance considerations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4.