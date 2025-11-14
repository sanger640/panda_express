#!/bin/bash

# Quick Start Script for Quest 3 Robot Teleoperation
# Run this script to check your setup

echo "================================"
echo "Quest 3 Robot Teleoperation Setup Checker"
echo "================================"
echo ""

# Check if conda environment exists
echo "Checking conda environment..."
if conda env list | grep -q "polymetis-local"; then
    echo "✓ polymetis-local environment found"
else
    echo "✗ polymetis-local environment not found"
    echo "  Create it with: conda env create -f ./polymetis/environment.yml"
fi
echo ""

# Check for required Python packages
echo "Checking Python packages..."
conda activate polymetis-local 2>/dev/null || source activate polymetis-local 2>/dev/null

python -c "import websockets" 2>/dev/null && echo "✓ websockets installed" || echo "✗ websockets missing (run: pip install websockets)"
python -c "import torch" 2>/dev/null && echo "✓ torch installed" || echo "✗ torch missing"
python -c "import polymetis" 2>/dev/null && echo "✓ polymetis installed" || echo "✗ polymetis missing"
echo ""

# Get IP address
echo "Your computer's IP addresses:"
if command -v hostname &> /dev/null; then
    hostname -I 2>/dev/null || ipconfig getifaddr en0 2>/dev/null || echo "Run 'ipconfig' on Windows"
fi
echo ""

# Check if files exist
echo "Checking required files..."
[ -f "quest_teleop.py" ] && echo "✓ quest_teleop.py found" || echo "✗ quest_teleop.py missing"
[ -f "quest_controller.html" ] && echo "✓ quest_controller.html found" || echo "✗ quest_controller.html missing"
echo ""

# Check if ports are available
echo "Checking if ports are available..."
if command -v netstat &> /dev/null; then
    if netstat -an 2>/dev/null | grep -q ":8765"; then
        echo "⚠ Port 8765 is in use"
    else
        echo "✓ Port 8765 is available"
    fi

    if netstat -an 2>/dev/null | grep -q ":8000"; then
        echo "⚠ Port 8000 is in use"
    else
        echo "✓ Port 8000 is available"
    fi
fi
echo ""

echo "================================"
echo "Setup Checker Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Update IP address in quest_controller.html"
echo "2. Start Polymetis: launch_robot.py robot_client=bullet_sim use_real_time=false gui=false"
echo "3. Start web server: python -m http.server 8000"
echo "4. Start bridge: python quest_teleop.py"
echo "5. Open Quest browser to: http://YOUR_IP:8000/quest_controller.html"
