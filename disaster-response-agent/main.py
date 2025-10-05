import asyncio
import uvicorn
import threading
import subprocess
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
load_dotenv()

def run_mcp_server():
    """Run the MCP Gateway server"""
    try:
        from src.orchestrator.mcp_server import app
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
    except Exception as e:
        print(f"Error starting MCP server: {e}")

def run_streamlit_dashboard():
    """Run the Streamlit dashboard"""
    try:
        subprocess.run([
            "streamlit", "run", 
            "src/ui/dashboard.py", 
            "--server.port", "8502",
            "headless", "true",
            "--server.address", "localhost",

        ])
    except Exception as e:
        print(f"Error starting Streamlit dashboard: {e}")

def setup_environment():
    """Setup initial environment"""
    print("🚨 Disaster Response Multi-Modal Agent Starting...")
    print("=" * 60)
    
    # Check for required environment variables
    required_vars = [
        "CEREBRAS_API_KEY",
        "GEMINI_API_KEY", 
        "HUGGINGFACE_TOKEN"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Warning: Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nCreate a .env file with your API keys or set environment variables.")
        print("See .env.example for the required format.\n")
    
    # Create necessary directories
    directories = [
        "data/temp_images",
        "data/datasets/sample_images",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup complete")
    print(f"📊 Dashboard will be available at: http://localhost:8501")
    print(f"🔗 API Gateway will be available at: http://localhost:8080")

    print("=" * 60)

if __name__ == "__main__":
    setup_environment()
    
    # Start MCP server in background thread
    mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
    mcp_thread.start()
    
    # Give MCP server time to start
    import time
    time.sleep(3)
    
    # Run Streamlit dashboard in main thread
    run_streamlit_dashboard()