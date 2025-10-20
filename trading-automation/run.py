#!/usr/bin/env python3
"""
Main runner script for trading automation system
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_utils import get_database_stats, update_equity_table_schema

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_menu():
    """Show the main menu"""
    print("\n" + "="*60)
    print("TRADING AUTOMATION SYSTEM")
    print("="*60)
    print("1. Preload existing Excel data to database")
    print("2. Run daily automation (fetch new emails)")
    print("3. Generate Excel file from database")
    print("4. Update Wpac values from date")
    print("5. Show database statistics")
    print("6. Exit")
    print("="*60)

def check_database_status():
    """Check if database has data"""
    try:
        # Update schema first
        update_equity_table_schema()
        stats = get_database_stats()
        return stats and stats['total_records'] > 0
    except:
        return False

def main():
    """Main function"""
    while True:
        show_menu()
        
        # Check database status
        has_data = check_database_status()
        if has_data:
            print("✓ Database is ready")
        else:
            print("⚠ Database is empty - you may need to run option 1 first")
        
        print("\nSelect an option (1-6): ", end="")
        
        try:
            choice = int(input())
        except ValueError:
            print("Invalid input. Please enter a number between 1-6.")
            continue
        
        if choice == 1:
            print("\nRunning preload script...")
            os.system(f"python {Path(__file__).parent}/scripts/preload_data.py")
            
        elif choice == 2:
            if not has_data:
                print("Database is empty. Please run option 1 first.")
                continue
            print("\nRunning daily automation...")
            os.system(f"python {Path(__file__).parent}/scripts/daily_automation.py")
            
        elif choice == 3:
            if not has_data:
                print("Database is empty. Please run option 1 first.")
                continue
            print("\nGenerating Excel file...")
            os.system(f"python {Path(__file__).parent}/scripts/generate_excel.py")
            
        elif choice == 4:
            if not has_data:
                print("Database is empty. Please run option 1 first.")
                continue
            print("\nRunning Wpac values update...")
            os.system(f"python {Path(__file__).parent}/scripts/update_wpac_values.py")
            
        elif choice == 5:
            print("\nDatabase statistics:")
            stats = get_database_stats()
            if stats:
                print(f"  Total records: {stats['total_records']}")
                print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
                if stats['latest_record']:
                    print(f"  Latest record: {stats['latest_record'][0]} (Equity: {stats['latest_record'][2]})")
            else:
                print("  No data found")
                
        elif choice == 6:
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please select 1-6.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()