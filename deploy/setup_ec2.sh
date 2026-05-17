#!/bin/bash
# ============================================================
# Prophet Forecasting Agent — AWS EC2 Deployment Script
# Run this on a fresh Ubuntu 22.04 EC2 instance (t3.small)
# ============================================================

set -e

echo "=========================================="
echo "  Prophet Oracle — EC2 Setup"
echo "=========================================="

# 1. System updates
echo "[1/7] Updating system..."
sudo apt update && sudo apt upgrade -y

# 2. Install Python 3.11
echo "[2/7] Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-venv python3.11-dev git curl

# 3. Clone the repo
echo "[3/7] Cloning repository..."
cd /home/ubuntu
git clone https://github.com/Shyamistic/TheProphetOracle.git
cd TheProphetOracle

# 4. Setup virtual environment
echo "[4/7] Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Create .env file (YOU MUST EDIT THIS)
echo "[5/7] Creating .env file..."
cat > .env << 'ENVEOF'
# === API Keys (Required) — REPLACE WITH YOUR ACTUAL KEYS ===
PROPHET_ANTHROPIC_API_KEY=YOUR_OPENROUTER_KEY_HERE
PROPHET_TAVILY_API_KEY=YOUR_TAVILY_KEY_HERE

# === Prophet Arena ===
PA_API_KEY=YOUR_PROPHET_ARENA_KEY_HERE
ANTHROPIC_API_KEY=YOUR_OPENROUTER_KEY_HERE

# === Optional ===
PROPHET_OPENAI_API_KEY=YOUR_OPENROUTER_KEY_HERE
PROPHET_FEATHERLESS_API_KEY=YOUR_FEATHERLESS_KEY_HERE
PROPHET_SERPER_API_KEY=YOUR_SERPER_KEY_HERE

# === Ensemble Models ===
PROPHET_ENSEMBLE_MODEL_1=anthropic/claude-sonnet-4
PROPHET_ENSEMBLE_MODEL_2=google/gemini-3.1-pro-preview
PROPHET_ENSEMBLE_MODEL_3=openai/gpt-5
PROPHET_FEATHERLESS_MODEL=Qwen/Qwen2.5-72B-Instruct

# === Server ===
PROPHET_PORT=8080
PROPHET_HOST=0.0.0.0

# === Calibration ===
PROPHET_SHRINKAGE_FACTOR=0.10
PROPHET_PLATT_COEFFICIENT=1.5
PROPHET_PER_EVENT_TIMEOUT_SECONDS=540
ENVEOF

# 6. Create systemd service (auto-restart on crash)
echo "[6/7] Creating systemd service..."
sudo tee /etc/systemd/system/prophet-agent.service > /dev/null << 'SERVICEEOF'
[Unit]
Description=Prophet Forecasting Agent
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/TheProphetOracle
Environment=PATH=/home/ubuntu/TheProphetOracle/venv/bin:/usr/bin
ExecStart=/home/ubuntu/TheProphetOracle/venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo systemctl daemon-reload
sudo systemctl enable prophet-agent
sudo systemctl start prophet-agent

# 7. Verify
echo "[7/7] Verifying..."
sleep 5
curl -s http://localhost:8080/health | python3.11 -m json.tool

echo ""
echo "=========================================="
echo "  DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "  Service: sudo systemctl status prophet-agent"
echo "  Logs:    sudo journalctl -u prophet-agent -f"
echo "  Health:  curl http://localhost:8080/health"
echo ""
echo "  Public URL: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
echo ""
echo "  Submit at: https://www.prophethacks.com/submit-endpoint"
echo "=========================================="
