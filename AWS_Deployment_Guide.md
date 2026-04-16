# AWS Deployment Guide: FCL Forecasting Dashboard

This document provides a step-by-step technical guide for deploying the FCL Forecasting Streamlit dashboard to an AWS EC2 instance, securing it behind a reverse proxy with SSL, and setting up automated CI/CD deployments via GitHub Actions.

## 1. AWS Infrastructure Setup

### Provision the EC2 Instance
1. Launch a new EC2 instance in your preferred AWS region.
2. **AMI:** Select Ubuntu Server 24.04 LTS (64-bit x86).
3. **Instance Type:** `t3.medium` (2 vCPUs, 4 GiB RAM) is recommended. The forecasting models and pandas dataframes require sufficient memory to load without crashing.
4. **Storage:** 20 GB General Purpose SSD (gp3).
5. **Key Pair:** Create or select an existing SSH key pair to access the instance.

### Configure the Security Group
The instance needs specific ports open to function correctly and securely.
- **Port 22 (SSH):** Restrict to your corporate IP or VPN range.
- **Port 80 (HTTP):** Open to 0.0.0.0/0 (required for Let's Encrypt SSL certificate generation and HTTP-to-HTTPS redirection).
- **Port 443 (HTTPS):** Restrict to your corporate IP/VPN range if the dashboard should only be accessible internally, or 0.0.0.0/0 if public access is required.

## 2. Server Configuration

SSH into the newly created EC2 instance using your key pair:
```bash
ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
```

### Install Dependencies
Update the system and install the required system packages:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv nginx git certbot python3-certbot-nginx
```

### Clone the Repository
Generate an SSH deploy key on the EC2 instance and add it to your GitHub repository (Settings > Deploy keys) to allow the server to pull code securely.
```bash
ssh-keygen -t ed25519 -C "aws-ec2-deploy"
cat ~/.ssh/id_ed25519.pub
```

Clone the repository into the ubuntu user's home directory:
```bash
git clone git@github.com:Alex8338/FCL-forecasting-model.git
cd FCL-forecasting-model
```

### Set Up the Python Environment
Create an isolated virtual environment and install the required Python packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Transfer Model Files
The large model bundle files (`.pkl`) are excluded from GitHub. You must transfer them manually to the EC2 instance once.
Using `scp` from your local machine:
```bash
scp -i /path/to/key.pem path1v2_final_models.pkl ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/FCL-forecasting-model/models/antarctica/
scp -i /path/to/key.pem path2_final_models.pkl ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/FCL-forecasting-model/models/antarctica/
```

## 3. Running Streamlit as a Background Service

To ensure the dashboard runs continuously and restarts automatically if the server reboots, configure it as a `systemd` service.

Create a new service file:
```bash
sudo nano /etc/systemd/system/fcl-dashboard.service
```

Add the following configuration:
```ini
[Unit]
Description=FCL Forecasting Streamlit Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/FCL-forecasting-model
Environment="PATH=/home/ubuntu/FCL-forecasting-model/venv/bin"
ExecStart=/home/ubuntu/FCL-forecasting-model/venv/bin/streamlit run app_antarctica.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable fcl-dashboard
sudo systemctl start fcl-dashboard
```

## 4. Reverse Proxy and SSL (Nginx)

Streamlit is currently running on port 8501 bound to `localhost`. We will use Nginx to securely expose it on port 443 (HTTPS) and map it to your custom domain.

### Configure Nginx
Create a new Nginx server block:
```bash
sudo nano /etc/nginx/sites-available/fcl-dashboard
```

Add the following configuration, replacing `fcl.yourdomain.com` with your actual domain:
```nginx
server {
    listen 80;
    server_name fcl.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (required by Streamlit)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

Enable the configuration and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/fcl-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Apply SSL Certificate
Run Certbot to automatically generate a free Let's Encrypt SSL certificate and configure HTTPS:
```bash
sudo certbot --nginx -d fcl.yourdomain.com
```

## 5. Automated CI/CD (GitHub Actions)

To allow Manus (or any developer) to push code to GitHub and have the AWS server update automatically, configure a GitHub Actions workflow.

### Add GitHub Secrets
In your GitHub repository, go to **Settings > Secrets and variables > Actions** and add:
- `EC2_HOST`: The public IP or domain of your EC2 instance.
- `EC2_USERNAME`: `ubuntu`
- `EC2_SSH_KEY`: The private SSH key (`.pem` file content) used to access the instance.

### Create the Workflow File
In the repository, create the following file at `.github/workflows/deploy.yml`:

```yaml
name: Deploy to AWS EC2

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to EC2
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USERNAME }}
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            cd /home/ubuntu/FCL-forecasting-model
            git pull origin main
            source venv/bin/activate
            pip install -r requirements.txt
            sudo systemctl restart fcl-dashboard
```

Once this file is committed and pushed to the `main` branch, every subsequent push will automatically trigger the workflow, pull the latest code on the EC2 instance, and restart the Streamlit service.
