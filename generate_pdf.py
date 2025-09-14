#!/usr/bin/env python3
"""
Standalone PDF Report Generator
==============================

This script generates a PDF report from existing analysis results.
Run this after your analysis is complete to create a professional PDF report.

Usage: python generate_pdf.py
"""

import sys
from pathlib import Path
from pdf_report_generator import generate_pdf_report

def main():
    """
    Generate PDF report from existing analysis results
    """
    print("📄 Standalone PDF Report Generator")
    print("=" * 40)
    
    output_dir = "output_plots"
    
    # Check if output directory exists
    if not Path(output_dir).exists():
        print(f"❌ Output directory '{output_dir}' not found!")
        print("💡 Make sure you've run the analysis first")
        return False
    
    # Check if required files exist
    required_files = [
        "performance_metrics_*.csv",
        "trade_log_*.csv"
    ]
    
    missing_files = []
    for pattern in required_files:
        if not list(Path(output_dir).glob(pattern)):
            missing_files.append(pattern)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        print("💡 Make sure you've run the complete analysis first")
        return False
    
    # Generate PDF
    try:
        pdf_path = generate_pdf_report(output_dir)
        
        if pdf_path:
            print(f"✅ PDF report generated successfully!")
            print(f"📄 Location: {pdf_path}")
            print(f"📁 Size: {pdf_path.stat().st_size / 1024:.1f} KB")
            return True
        else:
            print("❌ Failed to generate PDF report")
            return False
            
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)