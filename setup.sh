#!/bin/bash

# Deep Research Agent - Setup Script
# This script sets up the complete environment

set -e  # Exit on error

echo "=================================================="
echo "Deep Research Agent - Setup Script"
echo "=================================================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "❌ Error: Python 3.9 or higher required"
    exit 1
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ Pip upgraded"

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt
echo "✓ All dependencies installed"

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p data outputs/reports logs
echo "✓ Directories created"

# Check .env file
echo ""
echo "⚙️  Checking environment configuration..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found"
    echo ""
    echo "Please create .env file with your API keys:"
    echo ""
    echo "  SEC_USER_AGENT=YourName your@email.com"
    echo "  GOOGLE_API_KEY=your_gemini_api_key"
    echo "  TAVILY_API_KEY=your_tavily_api_key"
    echo ""
    echo "You can copy .env.example as a template:"
    echo "  cp .env.example .env"
    echo ""
else
    echo "✓ .env file found"
    
    # Validate required keys
    missing_keys=0
    
    if ! grep -q "^SEC_USER_AGENT=" .env; then
        echo "⚠️  Missing: SEC_USER_AGENT"
        missing_keys=1
    fi
    
    if ! grep -q "^GOOGLE_API_KEY=" .env; then
        echo "⚠️  Missing: GOOGLE_API_KEY"
        missing_keys=1
    fi
    
    if ! grep -q "^TAVILY_API_KEY=" .env; then
        echo "⚠️  Missing: TAVILY_API_KEY"
        missing_keys=1
    fi
    
    if [ $missing_keys -eq 0 ]; then
        echo "✓ All required keys present"
    else
        echo ""
        echo "⚠️  Please add missing keys to .env file"
    fi
fi

# Summary
echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment (if not already):"
echo "   source venv/bin/activate"
echo ""
echo "2. Configure API keys in .env file"
echo ""
echo "3. Test the setup:"
echo "   python test_workflow.py"
echo ""
echo "4. Run the Streamlit app:"
echo "   streamlit run app.py"
echo ""
echo "=================================================="
