#!/usr/bin/env python3
"""
Demo script for AI Attendance Admin Website
Demonstrates the complete admin website functionality
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 80)
    print("🤖 AI ATTENDANCE SYSTEM - ADMIN WEBSITE DEMO")
    print("=" * 80)
    print()
    print("🎯 This demo showcases the complete admin website for managing")
    print("   the AI-powered attendance system with the following features:")
    print()
    print("   ✅ Real-time Dashboard with live statistics")
    print("   ✅ Employee Management (Add/Edit/Delete)")
    print("   ✅ Check-in Monitoring with AI validation")
    print("   ✅ Reports & Analytics")
    print("   ✅ AI-powered insights and predictions")
    print("   ✅ System Settings and Configuration")
    print()

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if we're in the right directory
    if not os.path.exists('admin_website'):
        print("❌ admin_website directory not found!")
        print("   Please run this script from the project root directory.")
        return False
    
    # Check if admin files exist
    required_files = [
        'admin_website/index.html',
        'admin_website/styles.css', 
        'admin_website/script.js',
        'admin_website/admin_api.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found!")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn[standard]',
        'python-multipart',
        'pydantic'
    ]
    
    try:
        for package in required_packages:
            print(f"   Installing {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package, '--quiet'
            ])
        
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_admin_server():
    """Start the admin API server"""
    print("🚀 Starting admin API server...")
    
    # Change to admin_website directory
    os.chdir('admin_website')
    
    try:
        # Start the server in background
        process = subprocess.Popen([
            sys.executable, 'admin_api.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        if process.poll() is None:
            print("✅ Admin API server started successfully!")
            print("   🌐 Server running on: http://localhost:8001")
            print("   📚 API Documentation: http://localhost:8001/docs")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start server: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def open_website():
    """Open the admin website in browser"""
    print("🌐 Opening admin website in your browser...")
    
    try:
        webbrowser.open('http://localhost:8001')
        print("✅ Admin website opened in your default browser!")
        print()
        print("🎯 You can now explore the following features:")
        print("   • Dashboard: Real-time statistics and insights")
        print("   • Employee Management: Add, edit, delete employees")
        print("   • Check-ins: Monitor attendance with AI validation")
        print("   • Reports: Generate comprehensive reports")
        print("   • Analytics: AI-powered insights and predictions")
        print("   • Settings: Configure system parameters")
        return True
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        print("   Please manually open: http://localhost:8001")
        return False

def show_demo_instructions():
    """Show demo instructions"""
    print()
    print("📋 DEMO INSTRUCTIONS:")
    print("=" * 50)
    print()
    print("1. 🏠 DASHBOARD:")
    print("   • View real-time statistics")
    print("   • Check recent check-ins")
    print("   • Monitor employee status")
    print("   • Read AI insights")
    print()
    print("2. 👥 EMPLOYEE MANAGEMENT:")
    print("   • Click 'Add Employee' to register new employees")
    print("   • Use edit/delete buttons to manage existing employees")
    print("   • Upload employee photos for face recognition")
    print()
    print("3. 🕐 CHECK-IN MONITORING:")
    print("   • View all check-in records")
    print("   • Filter by date, employee, or anomalies")
    print("   • Monitor AI confidence scores")
    print("   • Detect fraud attempts")
    print()
    print("4. 📊 REPORTS & ANALYTICS:")
    print("   • Generate attendance reports")
    print("   • View department breakdowns")
    print("   • Analyze performance metrics")
    print("   • Export data for external analysis")
    print()
    print("5. 🧠 AI ANALYTICS:")
    print("   • View turnover risk analysis")
    print("   • Check attendance pattern analysis")
    print("   • Read predictive insights")
    print("   • Monitor anomaly detection")
    print()
    print("6. ⚙️ SETTINGS:")
    print("   • Configure geofences")
    print("   • Adjust AI model parameters")
    print("   • Set notification preferences")
    print("   • Manage system settings")
    print()

def main():
    """Main demo function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    print()
    print("🎬 Starting the demo...")
    print()
    
    # Start the admin server
    server_process = start_admin_server()
    if not server_process:
        return 1
    
    try:
        # Open the website
        open_website()
        
        # Show demo instructions
        show_demo_instructions()
        
        print("🎉 Demo is now running!")
        print("   Press Ctrl+C to stop the demo when you're done.")
        print()
        
        # Keep the server running
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 Stopping demo...")
        server_process.terminate()
        server_process.wait()
        print("✅ Demo stopped successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        server_process.terminate()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
