#!/bin/bash

# Simple RAG Chatbot Deployment Script
# Usage: ./deploy.sh [your-openai-api-key]

set -e

OPENAI_KEY=${1}

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Deploying RAG Chatbot...${NC}"

# Check if OpenAI key is provided
if [ -z "$OPENAI_KEY" ]; then
    echo -e "${RED}❌ OpenAI API key required!${NC}"
    echo "Usage: ./deploy.sh your_openai_api_key"
    exit 1
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Docker Compose...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Setup environment
echo -e "${GREEN}⚙️ Setting up environment...${NC}"
echo "OPENAI_API_KEY=${OPENAI_KEY}" > .env
echo "OPENAI_API_KEY=${OPENAI_KEY}" > backend/.env
echo "REACT_APP_API_URL=http://$(curl -s ifconfig.me):8000" > frontend/.env

# Setup firewall
echo -e "${GREEN}🔥 Configuring firewall...${NC}"
sudo ufw --force enable
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 8000/tcp

# Deploy
echo -e "${GREEN}🐳 Building and starting containers...${NC}"
docker-compose down 2>/dev/null || true
docker-compose up -d --build

# Wait for services
sleep 10

# Check status
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    echo ""
    echo "🌐 Your RAG Chatbot is running at:"
    echo "   Frontend: http://$(curl -s ifconfig.me)"
    echo "   Backend API: http://$(curl -s ifconfig.me):8000"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs: docker-compose logs"
    echo "   Restart: docker-compose restart"
    echo "   Stop: docker-compose down"
else
    echo -e "${RED}❌ Deployment failed. Check logs: docker-compose logs${NC}"
    exit 1
fi