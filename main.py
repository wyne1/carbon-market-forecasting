#!/usr/bin/env python3
"""
Main entry point for Carbon Market Forecasting
===============================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_app():
    """Run the Streamlit application"""
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"])

def run_app_with_transformer():
    """Run the Streamlit application"""
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app_with_transformer.py"])

def run_analysis(date=None):
    """Run the analysis pipeline"""
    from src.run_analysis import main
    main(date=date)

def main(): 
    """Main entry point"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Carbon Market Forecasting')
    parser.add_argument('command', choices=['app', 'analysis', 'app_with_transformer'], 
                       help='Command to run (app or analysis or app_with_transformer)')
    parser.add_argument('--date', type=str, 
                       help='Specific date for analysis (format: YYYY-MM-DD). If not provided, uses current date.')
    
    args = parser.parse_args()
    
    # Validate date format if provided
    if args.date:
        try:
            datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print("❌ Error: Invalid date format. Please use YYYY-MM-DD format (e.g., 2024-01-15)")
            sys.exit(1)
    
    if args.command == 'app':
        run_app()
    elif args.command == 'analysis':
        run_analysis(date=args.date)
    elif args.command == 'app_with_transformer':
        run_app_with_transformer()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python main.py [app|analysis|app_with_transformer] [--date YYYY-MM-DD]")
        print("  app      - Run the Streamlit application")
        print("  analysis - Run the analysis pipeline")
        print("  app_with_transformer - Run the Streamlit application with transformer")
        print("  --date   - Optional: Specific date for analysis (format: YYYY-MM-DD)")
        print("")
        print("Examples:")
        print("  python main.py analysis")
        print("  python main.py analysis --date 2024-01-15")
    else:
        main()
