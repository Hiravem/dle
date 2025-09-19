#!/usr/bin/env python3
"""
Startup script for AI Attendance Admin Website
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['fastapi', 'uvicorn', 'pydantic']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("Dependencies installed successfully!")
    
    return True

def start_admin_server():
    """Start the admin API server"""
    print("🚀 Starting AI Attendance Admin Website...")
    
    # Check if we're in the right directory
    if not os.path.exists('admin_api.py'):
        print("❌ admin_api.py not found. Please run this script from the admin_website directory.")
        return False
    
    try:
        # Start the FastAPI server
        print("📡 Starting API server on http://localhost:8001...")
        print("🌐 Opening admin website in your browser...")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:8001')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start the server
        import uvicorn
        uvicorn.run(
            "admin_api:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Admin server stopped.")
        return True
    except Exception as e:
        print(f"❌ Error starting admin server: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("🤖 AI ATTENDANCE SYSTEM - ADMIN WEBSITE")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return 1
    
    print("✅ All dependencies are ready!")
    print()
    
    # Start the server
    if start_admin_server():
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
