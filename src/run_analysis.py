#!/usr/bin/env python3
"""
Simple Runner Script for Carbon Market Analysis
==============================================

Usage: python run_analysis.py

This script will:
1. Load configuration from config.py
2. Run the complete analysis pipeline
3. Save all plots and results to the output folder

To customize the analysis, edit config.py instead of this file.
"""

import sys
from pathlib import Path
from src.automated_pipeline import AutomatedPipeline

try:
    from config.config import get_config
except ImportError:
    print("❌ Error: config.py not found!")
    print("💡 Make sure config.py is in the same directory as this script")
    sys.exit(1)

def main(date=None):
    """
    Main execution function
    """
    print("🚀 Carbon Market Analysis - Simple Runner")
    print("=" * 50)
    
    # Load configuration
    config = get_config()
    
    # Add custom date to config if provided
    if date:
        from datetime import datetime
        try:
            # Parse and validate the date
            custom_date = datetime.strptime(date, '%Y-%m-%d')
            config['custom_date'] = custom_date
            print(f"📅 Using custom date: {date}")
        except ValueError:
            print(f"❌ Error: Invalid date format '{date}'. Please use YYYY-MM-DD format.")
            sys.exit(1)
    else:
        print("📅 Using current date")
    
    print("📋 Configuration loaded:")
    print(f"   • Initial Balance: ${config['initial_balance']:,.2f}")
    print(f"   • Take Profit: {config['take_profit']*100:.1f}%")
    print(f"   • Stop Loss: {config['stop_loss']*100:.1f}%")
    print(f"   • Position Size: {config['position_size_fraction']*100:.0f}%")
    print(f"   • Max Epochs: {config['max_epochs']}")
    print(f"   • Output Directory: {config['output_dir']}")
    
    # Create and run pipeline
    try:
        pipeline = AutomatedPipeline(config=config)
        success = pipeline.run_full_pipeline()
        
        if success:
            print(f"\n🎊 SUCCESS! All results saved to: {Path(config['output_dir']).absolute()}")
            print("\n📁 Generated files:")
            output_dir = Path(config['output_dir'])
            for file_path in sorted(output_dir.glob("*")):
                if file_path.is_file():
                    print(f"   • {file_path.name}")
        else:
            print("\n💥 FAILED! Check error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()