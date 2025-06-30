# RAG Chatbot Makefile
# Simplifies common development and deployment tasks

.PHONY: help dev build deploy clean logs test lint

# Default target
help:
	@echo "RAG Chatbot - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  dev          - Start development environment"
	@echo "  build        - Build all containers"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy       - Deploy to production"
	@echo "  deploy-ssl   - Deploy with SSL"
	@echo "  restart      - Restart all services"
	@echo ""
	@echo "Maintenance:"
	@echo "  logs         - View application logs"
	@echo "  clean        - Clean up containers and volumes"
	@echo "  backup       - Backup application data"
	@echo "  restore      - Restore from backup"
	@echo ""

# Development commands
dev:
	@echo "🚀 Starting development environment..."
	@cp backend/.env.example backend/.env 2>/dev/null || true
	@cp frontend/.env.example frontend/.env 2>/dev/null || true
	@echo "⚠️  Please edit backend/.env and frontend/.env with your configuration"
	@echo "   Backend: Add your OPENAI_API_KEY"
	@echo "   Frontend: Set REACT_APP_API_URL=http://localhost:8000"
	@echo ""
	@echo "Then run: make up"

up:
	@echo "🏗️  Building and starting containers..."
	docker-compose up -d --build
	@echo "✅ Application is running!"
	@echo "   Frontend: http://localhost"
	@echo "   Backend: http://localhost:8000"
	@echo "   API Docs: http://localhost:8000/docs"

build:
	@echo "🏗️  Building all containers..."
	docker-compose build

# Deployment commands
deploy:
	@echo "🚀 Deploying to production..."
	@if [ ! -f backend/.env ]; then echo "❌ backend/.env not found. Run 'make dev' first."; exit 1; fi
	@if [ ! -f frontend/.env ]; then echo "❌ frontend/.env not found. Run 'make dev' first."; exit 1; fi
	docker-compose down
	docker-compose up -d --build
	@make status

deploy-ssl:
	@echo "🔒 Deploying with SSL..."
	@read -p "Enter your domain: " domain; \
	 read -p "Enter your email: " email; \
	 ./scripts/setup-ssl.sh $$domain $$email

restart:
	@echo "🔄 Restarting all services..."
	docker-compose restart
	@make status

# Maintenance commands
logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

status:
	@echo "📊 Application Status:"
	@echo ""
	@echo "🐳 Containers:"
	@docker-compose ps
	@echo ""
	@echo "💾 Disk Usage:"
	@docker system df
	@echo ""
	@echo "📈 Resource Usage:"
	@docker stats --no-stream

clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f
	docker volume prune -f
	@echo "✅ Cleanup completed"

clean-all:
	@echo "🧹 Cleaning everything..."
	docker-compose down -v --rmi all
	docker system prune -af
	docker volume prune -f
	@echo "✅ Deep cleanup completed"

# Testing commands
test:
	@echo "🧪 Running tests..."
	@echo "Backend tests:"
	docker-compose exec backend python -m pytest tests/ -v
	@echo "Frontend tests:"
	docker-compose exec frontend npm test -- --coverage --watchAll=false

test-backend:
	docker-compose exec backend python -m pytest tests/ -v

test-frontend:
	docker-compose exec frontend npm test -- --coverage --watchAll=false

# Linting commands
lint:
	@echo "🔍 Running linting..."
	@echo "Backend (Python):"
	docker-compose exec backend python -m flake8 app/
	@echo "Frontend (JavaScript):"
	docker-compose exec frontend npm run lint

lint-fix:
	@echo "🔧 Fixing linting issues..."
	docker-compose exec backend python -m black app/
	docker-compose exec frontend npm run lint:fix

# Backup and restore
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	 docker-compose exec backend tar -czf /tmp/uploads_$$timestamp.tar.gz uploads/; \
	 docker cp $$(docker-compose ps -q backend):/tmp/uploads_$$timestamp.tar.gz backups/; \
	 docker-compose exec backend tar -czf /tmp/chroma_$$timestamp.tar.gz chroma_db/; \
	 docker cp $$(docker-compose ps -q backend):/tmp/chroma_$$timestamp.tar.gz backups/; \
	 echo "✅ Backup created: backups/uploads_$$timestamp.tar.gz and backups/chroma_$$timestamp.tar.gz"

restore:
	@echo "📁 Available backups:"
	@ls -la backups/ 2>/dev/null || echo "No backups found"
	@read -p "Enter backup timestamp (YYYYMMDD_HHMMSS): " timestamp; \
	 if [ -f "backups/uploads_$$timestamp.tar.gz" ]; then \
	   docker cp backups/uploads_$$timestamp.tar.gz $$(docker-compose ps -q backend):/tmp/; \
	   docker-compose exec backend tar -xzf /tmp/uploads_$$timestamp.tar.gz; \
	   echo "✅ Uploads restored"; \
	 fi; \
	 if [ -f "backups/chroma_$$timestamp.tar.gz" ]; then \
	   docker cp backups/chroma_$$timestamp.tar.gz $$(docker-compose ps -q backend):/tmp/; \
	   docker-compose exec backend tar -xzf /tmp/chroma_$$timestamp.tar.gz; \
	   echo "✅ ChromaDB restored"; \
	 fi

# Database operations
db-reset:
	@echo "⚠️  This will delete all uploaded documents and vector data!"
	@read -p "Are you sure? (y/N): " confirm; \
	 if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
	   docker-compose exec backend rm -rf uploads/* chroma_db/*; \
	   echo "✅ Database reset completed"; \
	 else \
	   echo "❌ Operation cancelled"; \
	 fi

# Monitoring
monitor:
	@echo "📊 Monitoring application..."
	@echo "Press Ctrl+C to stop monitoring"
	@while true; do \
	   clear; \
	   echo "=== RAG Chatbot Monitoring ==="; \
	   echo ""; \
	   make status; \
	   echo ""; \
	   echo "=== Recent Logs ==="; \
	   docker-compose logs --tail=10; \
	   sleep 5; \
	 done

# Security
security-scan:
	@echo "🔒 Running security scan..."
	@echo "Scanning Docker images..."
	@docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
	   -v $$(pwd):/tmp aquasec/trivy image rag-chatbot-deploy_backend
	@docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
	   -v $$(pwd):/tmp aquasec/trivy image rag-chatbot-deploy_frontend

# Development helpers
shell-backend:
	docker-compose exec backend /bin/bash

shell-frontend:
	docker-compose exec frontend /bin/sh

install-deps:
	@echo "📦 Installing development dependencies..."
	@echo "Backend dependencies:"
	docker-compose exec backend pip install -r requirements.txt
	@echo "Frontend dependencies:"
	docker-compose exec frontend npm install

update-deps:
	@echo "🔄 Updating dependencies..."
	docker-compose exec backend pip install --upgrade -r requirements.txt
	docker-compose exec frontend npm update

# CI/CD helpers
ci-test:
	@echo "🔄 Running CI tests..."
	docker-compose -f docker-compose.yml -f docker-compose.test.yml up --build --abort-on-container-exit
	docker-compose -f docker-compose.yml -f docker-compose.test.yml down

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "Backend API docs available at: http://localhost:8000/docs"
	@echo "Frontend documentation:"
	docker-compose exec frontend npm run build-storybook