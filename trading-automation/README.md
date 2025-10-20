# Trading Automation System

This system automates the process of fetching trading data from emails and maintaining a database with calculated columns.

## Features

- **Database Management**: PostgreSQL database to store trading data
- **Email Processing**: Automatically fetch and process trading emails from Gmail
- **Calculated Columns**: Automatically calculate FUM, Margin Utilisation, and Drawdown
- **Excel Generation**: Generate Excel files with all data including calculated columns
- **User Input**: Interactive prompts for Wpac BizOne and Wpac Cash Reserve values

## Folder Structure

```
trading-automation/
├── data/                          # Place your Excel file here
├── scripts/                       # Individual scripts
│   ├── preload_data.py           # Migrate existing Excel to database
│   ├── daily_automation.py       # Fetch new emails and update database
│   └── generate_excel.py         # Generate Excel from database
├── config.py                     # Configuration settings
├── database_utils.py             # Database utilities
├── gmail_utils.py                # Gmail API utilities
├── data_processing.py            # Data processing utilities
├── excel_generator.py            # Excel generation utilities
└── run.py                        # Main runner script
```

## Setup

1. **Place your Excel file** in the `data/` folder
2. **Set up Gmail API credentials** (if you want to fetch emails):
   - Create `credentials.json` in the parent directory
   - The system will create `token.json` on first run
3. **Install dependencies**:
   ```bash
   pip install pandas psycopg2-binary openpyxl
   ```

## Usage

### Option 1: Use the main runner (Recommended)
```bash
python run.py
```

This will show a menu with all options.

### Option 2: Run individual scripts

#### 1. Preload existing data
```bash
python scripts/preload_data.py
```
- Migrates your existing Excel file to the database
- Prompts for Wpac BizOne and Wpac Cash Reserve values
- Sets up the database schema

#### 2. Daily automation
```bash
python scripts/daily_automation.py
```
- Fetches new emails since the last date in database
- Prompts for updated Wpac values (or uses previous ones)
- Processes new data and updates database
- Generates updated Excel file

#### 3. Generate Excel
```bash
python scripts/generate_excel.py
```
- Generates Excel file from current database
- Useful when you just want to export data

## Calculated Columns

The system automatically calculates these columns:

- **FUM** = Cash + Wpac BizOne + Wpac Cash Reserve
- **Margin Utilisation** = Maintenance Margin / FUM
- **Drawdown (Total FUM)** = Open Trade Equity / FUM

## Configuration

Edit `config.py` to modify:
- Database connection settings
- Gmail API credentials paths
- Default Wpac values

## Database Schema

The system creates a table `equity_data` with all original columns plus:
- `wpac_bizone` - User-provided Wpac BizOne value
- `wpac_cash_reserve` - User-provided Wpac Cash Reserve value
- `fum` - Calculated FUM value
- `margin_utilisation` - Calculated margin utilisation
- `drawdown_total_fum` - Calculated drawdown

## Workflow

1. **First time setup**:
   - Run `preload_data.py` to migrate existing Excel data
   - Provide Wpac values when prompted

2. **Daily updates**:
   - Run `daily_automation.py` to fetch new emails
   - Update Wpac values if needed
   - System processes new data and generates Excel

3. **Export data**:
   - Run `generate_excel.py` to create Excel file from database

## Troubleshooting

- **Database connection issues**: Check PostgreSQL is running and credentials in `config.py`
- **Gmail API issues**: Ensure `credentials.json` is in the parent directory
- **Excel file not found**: Place your Excel file in the `data/` folder
- **Permission errors**: Make sure scripts are executable (`chmod +x scripts/*.py`)

## Notes

- The system handles duplicate data gracefully (won't insert duplicates)
- Wpac values are stored per record and can be updated for new data
- Excel output includes all original columns plus calculated ones
- All scripts include comprehensive logging