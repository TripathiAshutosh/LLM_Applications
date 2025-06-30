# Complete Deployment Guide for RAG Chatbot on DigitalOcean

This guide will walk you through deploying your RAG chatbot application on DigitalOcean from scratch.

## 📋 Prerequisites

- **DigitalOcean Account**: Sign up at [digitalocean.com](https://digitalocean.com)
- **OpenAI API Key**: Get one from [platform.openai.com](https://platform.openai.com)
- **Domain Name** (optional): For custom domain and SSL
- **Basic Terminal Knowledge**: Comfortable with command line

## 🚀 Quick Deployment (5 Minutes)

### Option 1: Automated Script

```bash
# On your DigitalOcean droplet
git clone https://github.com/yourusername/rag-chatbot-deploy.git
cd rag-chatbot-deploy
chmod +x deploy.sh
./deploy.sh localhost your_openai_api_key_here
```

### Option 2: Using Makefile

```bash
git clone https://github.com/yourusername/rag-chatbot-deploy.git
cd rag-chatbot-deploy
make dev
# Edit .env files with your configuration
make up
```

## 📖 Step-by-Step Deployment

### Step 1: Create DigitalOcean Droplet

1. **Login to DigitalOcean Dashboard**
2. **Create a New Droplet**
   - **Image**: Ubuntu 22.04 LTS
   - **Plan**: Basic ($12/month, 2GB RAM recommended)
   - **Region**: Choose closest to your users
   - **Authentication**: SSH Key (recommended) or Password
   - **Hostname**: `rag-chatbot` or your preferred name

3. **Wait for Droplet Creation** (1-2 minutes)

### Step 2: Connect to Your Droplet

```bash
# SSH into your droplet
ssh root@YOUR_DROPLET_IP

# Update the system
apt update && apt upgrade -y
```

### Step 3: Install Required Software

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Add user to docker group
usermod -aG docker root

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install other utilities
apt install -y git curl nano ufw
```

### Step 4: Configure Firewall

```bash
# Enable firewall
ufw enable

# Allow necessary ports
ufw allow OpenSSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 8000/tcp  # Backend API

# Check status
ufw status
```

### Step 5: Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-chatbot-deploy.git
cd rag-chatbot-deploy

# Create environment files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

### Step 6: Configure Environment Variables

```bash
# Edit backend configuration
nano backend/.env
```

Add your OpenAI API key:
```bash
OPENAI_API_KEY=your_actual_openai_api_key_here
```

```bash
# Edit frontend configuration
nano frontend/.env
```

For testing with IP:
```bash
REACT_APP_API_URL=http://YOUR_DROPLET_IP:8000
```

For production with domain:
```bash
REACT_APP_API_URL=https://yourdomain.com/api
```

### Step 7: Deploy the Application

```bash
# Create main environment file for docker-compose
echo "OPENAI_API_KEY=your_actual_openai_api_key_here" > .env

# Build and start containers
docker-compose up -d --build

# Check if everything is running
docker-compose ps
```

You should see output like:
```
NAME                          COMMAND                  SERVICE     STATUS      PORTS
rag-chatbot-deploy-backend-1  "uvicorn app.main:ap…"   backend     running     0.0.0.0:8000->8000/tcp
rag-chatbot-deploy-frontend-1 "docker-entrypoint.s…"   frontend    running     0.0.0.0:80->80/tcp
```

### Step 8: Test Your Application

```bash
# Test backend health
curl http://localhost:8000/health

# Test frontend
curl http://localhost
```

Visit `http://YOUR_DROPLET_IP` in your browser to access the application!

## 🔒 Production Setup with Custom Domain

### Step 1: Point Domain to Droplet

1. **Add A Record**: Point your domain to your droplet's IP
   ```
   Type: A
   Name: @ (or subdomain)
   Value: YOUR_DROPLET_IP
   TTL: 300
   ```

2. **Wait for DNS Propagation** (5-30 minutes)

### Step 2: Get SSL Certificate

```bash
# Install Certbot
apt install -y certbot python3-certbot-nginx

# Stop frontend container temporarily
docker-compose stop frontend

# Get SSL certificate
certbot certonly --standalone -d yourdomain.com

# Restart frontend
docker-compose start frontend
```

### Step 3: Configure Production Environment

```bash
# Update frontend environment for production
nano frontend/.env
```

Change to:
```bash
REACT_APP_API_URL=https://yourdomain.com/api
```

```bash
# Rebuild with new configuration
docker-compose up -d --build
```

### Step 4: Setup Nginx Reverse Proxy (Optional)

For advanced production setup with reverse proxy:

```bash
# Use production profile
export COMPOSE_PROFILES=production
docker-compose up -d

# Update nginx.conf with your domain
nano nginx.conf
# Replace 'your-domain.com' with your actual domain

# Restart nginx service
docker-compose restart nginx
```

## 🔧 Maintenance & Operations

### Viewing Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs backend
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f backend
```

### Updating the Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose up -d --build

# Check status
docker-compose ps
```

### Backup Data

```bash
# Create backup directory
mkdir -p /backups

# Backup uploads and vector database
docker-compose exec backend tar -czf /tmp/backup-$(date +%Y%m%d).tar.gz uploads/ chroma_db/
docker cp $(docker-compose ps -q backend):/tmp/backup-$(date +%Y%m%d).tar.gz /backups/
```

### Resource Monitoring

```bash
# Check Docker resource usage
docker stats

# Check disk space
df -h

# Check memory usage
free -h

# Check system load
htop
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **Port Already in Use**
```bash
# Find process using port 80
sudo lsof -i :80

# Stop Apache if running
sudo systemctl stop apache2
sudo systemctl disable apache2

# Or stop Nginx
sudo systemctl stop nginx
sudo systemctl disable nginx
```

#### 2. **Docker Permission Denied**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Logout and login again
exit
# SSH back in
```

#### 3. **Out of Memory**
```bash
# Check memory usage
free -h

# Restart services to free memory
docker-compose restart

# Consider upgrading droplet size
```

#### 4. **SSL Certificate Issues**
```bash
# Check certificate status
certbot certificates

# Renew certificate
certbot renew --dry-run

# Force renewal
certbot renew --force-renewal
```

#### 5. **Application Not Accessible**
```bash
# Check if containers are running
docker-compose ps

# Check container logs
docker-compose logs backend
docker-compose logs frontend

# Check firewall
ufw status

# Test internal connectivity
curl http://localhost:8000/health
curl http://localhost
```

#### 6. **File Upload Fails**
```bash
# Check upload directory permissions
docker-compose exec backend ls -la uploads/

# Check disk space
df -h

# Check backend logs for errors
docker-compose logs backend | grep -i error
```

### Performance Optimization

#### 1. **Increase Droplet Resources**
- Upgrade to 4GB+ RAM for production use
- Consider CPU-optimized droplets for heavy processing

#### 2. **Enable Swap** (for low memory droplets)
```bash
# Create 2GB swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### 3. **Docker Optimization**
```bash
# Clean up unused containers and images
docker system prune -a

# Limit container resources in docker-compose.yml
# Add to each service:
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '0.5'
```

## 📊 Monitoring and Alerts

### Basic Monitoring Setup

```bash
# Install monitoring tools
apt install -y htop iotop nethogs

# Create monitoring script
cat > /usr/local/bin/monitor.sh << 'EOF'
#!/bin/bash
echo "=== System Status ==="
date
echo "=== Memory Usage ==="
free -h
echo "=== Disk Usage ==="
df -h
echo "=== Docker Status ==="
docker-compose ps
echo "=== Recent Logs ==="
docker-compose logs --tail=20
EOF

chmod +x /usr/local/bin/monitor.sh

# Run monitoring
/usr/local/bin/monitor.sh
```

### Automated Health Checks

```bash
# Create health check script
cat > /usr/local/bin/health-check.sh << 'EOF'
#!/bin/bash
cd /root/rag-chatbot-deploy

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    echo "Services not running, restarting..."
    docker-compose restart
fi

# Check backend health
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "Backend unhealthy, restarting..."
    docker-compose restart backend
fi
EOF

chmod +x /usr/local/bin/health-check.sh

# Add to crontab (run every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/health-check.sh") | crontab -
```

## 🚦 Final Checklist

- [ ] Droplet created and accessible via SSH
- [ ] Docker and Docker Compose installed
- [ ] Firewall configured (ports 80, 443, 8000 open)
- [ ] Environment variables set (OpenAI API key)
- [ ] Application deployed and containers running
- [ ] Frontend accessible via browser
- [ ] Backend API responding to health checks
- [ ] File upload functionality tested
- [ ] Chat functionality working
- [ ] SSL certificate installed (if using custom domain)
- [ ] Backup strategy implemented
- [ ] Monitoring setup configured

## 🎉 Success!

Your RAG chatbot is now live! You can:

1. **Upload PDFs** through the web interface
2. **Chat with your documents** using natural language
3. **Manage multiple sessions** with different document sets
4. **Scale the application** by upgrading your droplet

## 📞 Support

If you encounter issues:

1. **Check the logs**: `docker-compose logs`
2. **Review this guide**: Most issues are covered in troubleshooting
3. **Check GitHub Issues**: Search for similar problems
4. **Create an Issue**: Include logs and detailed error information

## 🔗 Useful Commands Reference

```bash
# Start application
docker-compose up -d

# Stop application
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Clean up
docker system prune -f

# Backup data
make backup

# Monitor resources
docker stats
```

Happy deploying! 🚀