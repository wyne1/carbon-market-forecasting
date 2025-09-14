#!/bin/bash

echo "===================================================="
echo "   Carbon Market Analysis - Automated Pipeline"
echo "===================================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ ERROR: Python is not installed or not in PATH"
        echo "💡 Please install Python 3.8+ and try again"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "🐍 Using Python: $($PYTHON_CMD --version)"
echo

echo "🚀 Starting analysis..."
echo

# Run the analysis
$PYTHON_CMD run_analysis.py

# Check exit status
if [ $? -eq 0 ]; then
    echo
    echo "✅ Analysis completed successfully!"
    echo "📁 Check the output_plots folder for results."
else
    echo
    echo "❌ Analysis failed. Check error messages above."
    exit 1
fi

echo
echo "🎉 All done!"