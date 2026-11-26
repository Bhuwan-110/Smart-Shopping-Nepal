# SmartShopping Nepal — Production Deployment Guide

This guide details the complete deployment architecture, security hardening, database setup, and operational instructions for **Smart Shopping Nepal**.

---

## 1. Architecture Overview

Smart Shopping Nepal uses a coordinated dual-interface architecture:
- **Primary Customer-Facing Website (Django 5+)**:
  - Serves search, price comparison, categories, product details, user authentication, wishlists, and recommendations.
  - Served in production using **Gunicorn** behind an **Nginx** reverse proxy on port 80/443.
  - Relational database: **PostgreSQL** with connection pooling.
- **Internal Research & Analytics Dashboard (Streamlit)**:
  - Serves internal research, data collection monitoring, matching audits, and recommendation model evaluation.
  - Deployed as an internal service (e.g., port 8501) restricted to admin/internal VPN.
- **Python Data Pipeline (`src/`)**:
  - Scheduled batch processing jobs syncing partner feeds, manual imports, and TF-IDF similarity matrices.

---

## 2. Prerequisites & Environment Setup

### System Requirements
- Ubuntu 22.04 LTS or Debian 12 (or Windows Server 2022)
- Python 3.10+
- PostgreSQL 14+
- Nginx & Git

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <repository-url> /var/www/smartshoppingnepal
cd /var/www/smartshoppingnepal

# Create isolated Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Environment & Security Configuration

Copy the `.env.example` template to `.env`:
```bash
cp .env.example .env
chmod 600 .env
```

### Production `.env` Settings
```ini
DJANGO_SECRET_KEY="generate-a-strong-64-character-random-key"
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS="smartshoppingnepal.com,www.smartshoppingnepal.com,127.0.0.1"

# Database Configuration (PostgreSQL)
USE_POSTGRES=True
DB_NAME=smart_shopping_db
DB_USER=smartshop_user
DB_PASSWORD="your-strong-database-password"
DB_HOST=127.0.0.1
DB_PORT=5432

# Production HTTPS / Security Headers
SECURE_SSL_REDIRECT=True
CSRF_COOKIE_SECURE=True
SESSION_COOKIE_SECURE=True
SECURE_HSTS_SECONDS=31536000
```

---

## 4. PostgreSQL Database Setup

Log into PostgreSQL and create the production database and user:
```sql
CREATE DATABASE smart_shopping_db;
CREATE USER smartshop_user WITH ENCRYPTED PASSWORD 'your-strong-database-password';
GRANT ALL PRIVILEGES ON DATABASE smart_shopping_db TO smartshop_user;
ALTER DATABASE smart_shopping_db OWNER TO smartshop_user;
```

Apply database migrations:
```bash
python manage.py migrate
```

Create superuser account:
```bash
python manage.py createsuperuser
```

---

## 5. Pipeline Synchronization & Static Files

### Synchronize Products & Platforms into Database
```bash
python manage.py sync_pipeline
```

### Collect Static Files for Nginx
```bash
python manage.py collectstatic --noinput
```

---

## 6. Web Server & Process Management (Gunicorn + Systemd + Nginx)

### Systemd Service for Django (`/etc/systemd/system/smartshopping.service`)
```ini
[Unit]
Description=SmartShopping Nepal Gunicorn Daemon
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/smartshoppingnepal
ExecStart=/var/www/smartshoppingnepal/.venv/bin/gunicorn \
    --access-logfile /var/log/gunicorn/access.log \
    --error-logfile /var/log/gunicorn/error.log \
    --workers 3 \
    --bind 127.0.0.1:8000 \
    smart_shopping_web.wsgi:application

[Install]
WantedBy=multi-user.target
```

### Systemd Service for Streamlit Dashboard (`/etc/systemd/system/smartshopping-dashboard.service`)
```ini
[Unit]
Description=SmartShopping Nepal Streamlit Research Dashboard
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/smartshoppingnepal
ExecStart=/var/www/smartshoppingnepal/.venv/bin/streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy Configuration (`/etc/nginx/sites-available/smartshoppingnepal`)
```nginx
server {
    listen 80;
    server_name smartshoppingnepal.com www.smartshoppingnepal.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name smartshoppingnepal.com www.smartshoppingnepal.com;

    ssl_certificate /etc/letsencrypt/live/smartshoppingnepal.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/smartshoppingnepal.com/privkey.pem;

    # Static Assets
    location /static/ {
        alias /var/www/smartshoppingnepal/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }

    # Customer Website (Django)
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Internal Research Dashboard (Streamlit - IP Restricted)
    location /research-dashboard/ {
        # Allow internal office or VPN IPs only
        allow 192.168.1.0/24;
        deny all;

        proxy_pass http://127.0.0.1:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## 7. Verification & Health Monitoring

Run all automated test suites prior to production traffic cutover:
```bash
# Django and security integration tests
python manage.py test

# Data processing, matching, and Streamlit dashboard tests
python -m unittest discover -s tests
```

### Process Status
```bash
sudo systemctl enable --now smartshopping
sudo systemctl enable --now smartshopping-dashboard
sudo systemctl restart nginx
```
