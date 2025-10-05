# Deployment Guide - Disaster Response Multi-Modal Agent

## Overview

This guide provides step-by-step instructions for deploying the Disaster Response Multi-Modal Agent system in various environments, from local development to production-scale deployments.

## Prerequisites

### System Requirements

**Minimum Requirements (Development)**:
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **CPU**: 2 cores, 2.4 GHz
- **RAM**: 4 GB
- **Storage**: 10 GB free space
- **Python**: 3.9+

**Recommended Requirements (Production)**:
- **OS**: Ubuntu 22.04 LTS or CentOS 8+
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 100 GB SSD
- **Python**: 3.11+
- **Docker**: 20.10+

### Required Services

**Free API Keys** (Required):
- **Cerebras API**: https://cloud.cerebras.ai/ (Free tier available)
- **Google Gemini API**: https://makersuite.google.com/ (Free tier available)

**Optional Services**:
- PostgreSQL database (SQLite used by default)
- Redis for caching
- Nginx for reverse proxy

## Quick Start (5-Minute Setup)

### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/your-org/disaster-response-agent.git
cd disaster-response-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Minimal configuration**:
```bash
# Required API Keys
CEREBRAS_API_KEY=your_cerebras_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Generate a secret key
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Basic settings
HOST=0.0.0.0
PORT=8080
DEBUG=false
```

### 3. Initialize Database
```bash
# Load sample data for testing
python scripts/load_demo_data.py

# Run health check
python scripts/health_check.py
```

### 4. Start Services
```bash
# Start main application
python main.py
```

### 5. Access Dashboard
- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

## Docker Deployment

### Single Container
```bash
# Build and run
docker build -t disaster-response .
docker run -p 8080:8080 -p 8501:8501 \
  -e CEREBRAS_API_KEY=your_key \
  -e GEMINI_API_KEY=your_key \
  disaster-response
```

### Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Production compose override**:
```bash
# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Production Deployment

### 1. Server Preparation

#### Ubuntu 22.04 Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install system dependencies
sudo apt install -y nginx certbot python3-certbot-nginx fail2ban ufw

# Configure firewall
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw --force enable
```

#### CentOS/RHEL Setup
```bash
# Update system
sudo yum update -y

# Install Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker

# Install additional tools
sudo yum install -y nginx certbot python3-certbot-nginx epel-release
```

### 2. Application Deployment

#### Create Directory Structure
```bash
sudo mkdir -p /opt/disaster-response
sudo chown $USER:$USER /opt/disaster-response
cd /opt/disaster-response

# Clone repository
git clone https://github.com/your-org/disaster-response-agent.git .
```

#### Environment Configuration
```bash
# Create production environment file
cat > .env.production << EOF
# Production Configuration
ENVIRONMENT=production
DEBUG=false

# API Keys (set your actual keys)
CEREBRAS_API_KEY=your_production_cerebras_key
GEMINI_API_KEY=your_production_gemini_key
SECRET_KEY=$(openssl rand -base64 32)

# Database
DATABASE_URL=postgresql://disaster_user:secure_password@localhost:5432/disaster_response

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com

# Performance
WORKERS=4
MAX_WORKERS=8

# Monitoring
LOG_LEVEL=WARNING
SENTRY_DSN=your_sentry_dsn_here

# SSL
SSL_CERT_PATH=/etc/letsencrypt/live/your-domain.com/fullchain.pem
SSL_KEY_PATH=/etc/letsencrypt/live/your-domain.com/privkey.pem
EOF
```

#### Database Setup (Optional PostgreSQL)
```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Configure database
sudo -u postgres psql << EOF
CREATE DATABASE disaster_response;
CREATE USER disaster_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE disaster_response TO disaster_user;
\q
EOF

# Configure PostgreSQL
sudo nano /etc/postgresql/14/main/postgresql.conf
# Uncomment and modify:
# listen_addresses = 'localhost'
# max_connections = 100

sudo systemctl restart postgresql
```

#### Redis Setup (Optional)
```bash
# Install Redis
sudo apt install -y redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
# Modify these settings:
# maxmemory 256mb
# maxmemory-policy allkeys-lru

sudo systemctl restart redis-server
```

### 3. SSL Certificate Setup
```bash
# Install SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

### 4. Nginx Configuration
```bash
# Create Nginx config
sudo tee /etc/nginx/sites-available/disaster-response << EOF
upstream disaster_api {
    server 127.0.0.1:8080;
}

upstream disaster_dashboard {
    server 127.0.0.1:8501;
}

# API Server
server {
    listen 80;
    server_name api.your-domain.com;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # File upload limits
    client_max_body_size 50M;

    location / {
        proxy_pass http://disaster_api;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location /ws/ {
        proxy_pass http://disaster_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
    }
}

# Dashboard Server
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://disaster_dashboard;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /_stcore/stream {
        proxy_pass http://disaster_dashboard;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/disaster-response /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 5. Systemd Service Setup
```bash
# Create systemd service
sudo tee /etc/systemd/system/disaster-response.service << EOF
[Unit]
Description=Disaster Response Multi-Modal Agent
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=forking
User=www-data
Group=www-data
WorkingDirectory=/opt/disaster-response
Environment=PATH=/opt/disaster-response/venv/bin
EnvironmentFile=/opt/disaster-response/.env.production
ExecStart=/opt/disaster-response/venv/bin/python main.py --daemon
ExecReload=/bin/kill -HUP \$MAINPID
PIDFile=/opt/disaster-response/disaster-response.pid
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable disaster-response
sudo systemctl start disaster-response
```

## Cloud Deployment

### AWS EC2 Deployment

#### 1. Launch EC2 Instance
```bash
# Launch t3.large instance with Ubuntu 22.04
# Configure security group:
# - SSH (22): Your IP
# - HTTP (80): 0.0.0.0/0
# - HTTPS (443): 0.0.0.0/0
# - Custom (8080): 0.0.0.0/0  # API
# - Custom (8501): 0.0.0.0/0  # Dashboard
```

#### 2. Setup Script
```bash
#!/bin/bash
# AWS EC2 setup script

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and deploy
git clone https://github.com/your-org/disaster-response-agent.git
cd disaster-response-agent

# Set environment variables
export CEREBRAS_API_KEY="your_key"
export GEMINI_API_KEY="your_key"

# Deploy with Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Google Cloud Platform

#### 1. Cloud Run Deployment
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/disaster-response

# Deploy to Cloud Run
gcloud run deploy disaster-response \
    --image gcr.io/PROJECT_ID/disaster-response \
    --platform managed \
    --region us-central1 \
    --set-env-vars CEREBRAS_API_KEY=your_key,GEMINI_API_KEY=your_key \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10
```

#### 2. Kubernetes Deployment
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: disaster-response
spec:
  replicas: 3
  selector:
    matchLabels:
      app: disaster-response
  template:
    metadata:
      labels:
        app: disaster-response
    spec:
      containers:
      - name: disaster-response
        image: gcr.io/PROJECT_ID/disaster-response:latest
        ports:
        - containerPort: 8080
        - containerPort: 8501
        env:
        - name: CEREBRAS_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: cerebras-api-key
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: gemini-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: disaster-response-service
spec:
  selector:
    app: disaster-response
  ports:
  - name: api
    protocol: TCP
    port: 8080
    targetPort: 8080
  - name: dashboard
    protocol: TCP
    port: 8501
    targetPort: 8501
  type: LoadBalancer
```

### Azure Container Instances
```bash
# Create resource group
az group create --name disaster-response-rg --location eastus

# Deploy container
az container create \
    --resource-group disaster-response-rg \
    --name disaster-response \
    --image your-registry/disaster-response:latest \
    --ports 8080 8501 \
    --environment-variables \
        CEREBRAS_API_KEY=your_key \
        GEMINI_API_KEY=your_key \
    --memory 2 \
    --cpu 1
```

## Monitoring and Maintenance

### Health Monitoring Setup
```bash
# Create monitoring script
cat > /opt/disaster-response/monitor.sh << 'EOF'
#!/bin/bash
LOG_FILE="/var/log/disaster-response-monitor.log"

check_service() {
    if ! curl -f -s http://localhost:8080/health > /dev/null; then
        echo "[$(date)] API service is down - restarting" >> $LOG_FILE
        sudo systemctl restart disaster-response
    fi
    
    if ! curl -f -s http://localhost:8501 > /dev/null; then
        echo "[$(date)] Dashboard service is down - restarting" >> $LOG_FILE
        sudo systemctl restart disaster-response
    fi
}

check_service
EOF

chmod +x /opt/disaster-response/monitor.sh

# Add to crontab
echo "*/5 * * * * /opt/disaster-response/monitor.sh" | sudo crontab -
```

### Log Rotation Setup
```bash
# Create logrotate config
sudo tee /etc/logrotate.d/disaster-response << EOF
/opt/disaster-response/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 www-data www-data
    postrotate
        sudo systemctl reload disaster-response
    endscript
}
EOF
```

### Backup Configuration
```bash
# Create backup script
cat > /opt/disaster-response/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/disaster-response"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
if [ "$DATABASE_URL" != "sqlite:///data/disaster_response.db" ]; then
    pg_dump $DATABASE_URL | gzip > $BACKUP_DIR/database_$DATE.sql.gz
else
    cp /opt/disaster-response/data/disaster_response.db $BACKUP_DIR/database_$DATE.db
fi

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz -C /opt/disaster-response .env.production config/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*" -mtime +30 -delete

echo "Backup completed: $DATE"
EOF

chmod +x /opt/disaster-response/backup.sh

# Schedule daily backup
echo "0 2 * * * /opt/disaster-response/backup.sh" | sudo crontab -
```

## Scaling and Performance

### Horizontal Scaling
```yaml
# Docker Swarm scaling
version: '3.8'
services:
  disaster-response:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
```

### Load Balancer Configuration
```nginx
upstream disaster_response_api {
    least_conn;
    server 10.0.1.10:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8080 max_fails=3 fail_timeout=30s;
}
```

### Database Optimization
```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

SELECT pg_reload_conf();
```

## Security Hardening

### SSL/TLS Configuration
```nginx
# Strong SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 1d;
ssl_stapling on;
ssl_stapling_verify on;
```

### Firewall Configuration
```bash
# Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

### Fail2Ban Setup
```ini
# /etc/fail2ban/jail.local
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[nginx-http-auth]
enabled = true

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
action = iptables-multiport[name=ReqLimit, port="http,https", protocol=tcp]
logpath = /var/log/nginx/error.log
findtime = 600
bantime = 7200
maxretry = 10
```

## Troubleshooting

### Common Issues

#### API Service Not Starting
```bash
# Check logs
journalctl -u disaster-response -f

# Check port availability
sudo netstat -tulpn | grep :8080

# Check environment variables
sudo -u www-data env | grep -E "(CEREBRAS|GEMINI)"
```

#### Database Connection Issues
```bash
# Test database connection
python3 -c "
import os
from sqlalchemy import create_engine
engine = create_engine(os.getenv('DATABASE_URL'))
print('Database connection: OK' if engine.connect() else 'FAILED')
"
```

#### High Memory Usage
```bash
# Monitor memory usage
htop

# Check Python processes
ps aux | grep python | grep -v grep

# Restart service if needed
sudo systemctl restart disaster-response
```

### Performance Tuning
```bash
# Increase file limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 1024" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 1024" >> /etc/sysctl.conf
sysctl -p
```

## Maintenance Tasks

### Regular Maintenance Checklist
- [ ] Check service status: `systemctl status disaster-response`
- [ ] Review logs: `journalctl -u disaster-response --since "1 day ago"`
- [ ] Check disk usage: `df -h`
- [ ] Monitor memory usage: `free -h`
- [ ] Review error rates in dashboard
- [ ] Test API endpoints: `python scripts/health_check.py`
- [ ] Check SSL certificate expiry: `certbot certificates`
- [ ] Review security logs: `sudo grep "Failed password" /var/log/auth.log`

### Update Procedure
```bash
# 1. Backup current installation
./backup.sh

# 2. Pull latest updates
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt --upgrade

# 4. Run database migrations (if any)
python scripts/migrate_database.py

# 5. Restart service
sudo systemctl restart disaster-response

# 6. Verify health
python scripts/health_check.py
```

This deployment guide provides comprehensive instructions for setting up the Disaster Response Multi-Modal Agent system across various environments and scenarios.