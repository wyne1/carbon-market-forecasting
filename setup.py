from setuptools import setup, find_packages

setup(
    name="carbon-market-forecasting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tensorflow>=2.8.0",
        "scikit-learn>=0.24.0",
        "streamlit>=1.10.0",
        "pymongo>=3.12.0",
        "openpyxl>=3.0.0",
        "xlrd>=2.0.0",
        "plotly>=5.0.0",
    ],
    python_requires=">=3.8",
)
