from setuptools import setup, find_packages

setup(
    name="stacpred",
    version="1.0.0",
    author="Malaka Sandaruwan",
    description="Physics-Based Stacking Ensemble for SIRT1 Allosteric Affinity Prediction",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn==1.7.2",
        "xgboost>=2.0.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "rdkit>=2023.9.1"
    ],
)