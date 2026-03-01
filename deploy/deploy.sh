#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER="squall@180.152.71.166"
SSH_KEY="/Users/jialechen/.ssh/lab_deploy_180_152_71_166"
REMOTE_DIR="/opt/weld-sim"

echo "=== Building frontend ==="
cd "$PROJECT_DIR/frontend" && npm run build && cd "$PROJECT_DIR"

echo "=== Syncing to server ==="
rsync -avz \
  --exclude='.venv' --exclude='node_modules' --exclude='.git' \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='frontend/node_modules' --exclude='frontend/.vite' \
  --exclude='.claude' --exclude='data/' \
  -e "ssh -i $SSH_KEY" \
  "$PROJECT_DIR/" "$SERVER:$REMOTE_DIR/"

echo "=== Setting up server ==="
ssh -i "$SSH_KEY" "$SERVER" << 'REMOTE'
cd /opt/weld-sim
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -r requirements.txt -r requirements-web.txt -q
pip install -e backend/ -q 2>/dev/null || pip install sqlalchemy[asyncio] asyncpg alembic pydantic-settings celery redis reportlab openpyxl -q
mkdir -p data data/reports data/uploads data/logs data/storage
echo "=== Testing ==="
python3 -m pytest tests/test_web/test_health.py -v 2>&1 | tail -5
echo "=== Updating systemd service ==="
sudo cp deploy/weld-sim.service /etc/systemd/system/
sudo systemctl daemon-reload
echo "=== Restarting service ==="
if systemctl is-active --quiet weld-sim; then
    sudo systemctl restart weld-sim
    echo "Service restarted"
else
    sudo systemctl enable --now weld-sim
    echo "Service enabled and started"
fi
echo "=== Verifying ==="
sleep 2
echo "v1 API:"
curl -s http://localhost:8001/api/v1/health || echo "v1 not responding"
echo ""
echo "v2 API:"
curl -s http://localhost:8001/api/v2/health || echo "v2 not responding"
echo ""
echo "=== Done ==="
REMOTE
