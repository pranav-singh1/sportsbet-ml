#!/usr/bin/env python3
"""
Setup script for the Sports Betting ML System.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["data", "models", "logs", "results", "reports"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

def setup_environment():
    """Set up the environment."""
    print("Setting up Sports Betting ML System...")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if install_requirements():
        print("\n" + "="*50)
        print("SETUP COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nNext steps:")
        print("1. Add your API keys to config/config.py")
        print("2. Run the example: python3 example_usage.py")
        print("3. Run the main pipeline: python3 src/main.py")
        print("4. Explore the Jupyter notebooks")
        return True
    else:
        print("\n" + "="*50)
        print("SETUP FAILED!")
        print("="*50)
        print("Please check the error messages above and try again.")
        return False

if __name__ == "__main__":
    setup_environment()
