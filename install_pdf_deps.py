#!/usr/bin/env python3
"""
PDF Dependencies Installer
==========================

This script installs the required dependencies for PDF report generation.
Run this once before using the PDF features.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """
    Install PDF dependencies
    """
    print("📦 Installing PDF Report Dependencies")
    print("=" * 40)
    
    # Required packages for PDF generation
    packages = [
        "reportlab",
        "Pillow",  # For image handling
    ]
    
    failed_packages = []
    
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
            failed_packages.append(package)
    
    print("\n" + "=" * 40)
    
    if not failed_packages:
        print("🎉 All dependencies installed successfully!")
        print("📄 You can now use PDF report generation features")
        return True
    else:
        print("❌ Some packages failed to install:")
        for package in failed_packages:
            print(f"   • {package}")
        print("\n💡 Try installing manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)