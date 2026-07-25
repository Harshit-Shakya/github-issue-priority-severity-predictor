from setuptools import setup, find_packages

setup(
    name="github-issue-labeler",
    version="1.0.0",
    description="GitHub Issue Auto-Labeler with Priority Scoring",
    author="Ishan",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "datasets>=2.14.6",
        "transformers>=4.36.2",
        "tensorflow>=2.15.0",
        "scikit-learn>=1.3.2",
        "xgboost>=2.0.3",
        "textblob>=0.17.1",
        "shap>=0.43.0",
        "streamlit>=1.28.2",
        "plotly>=5.17.0",
        "requests>=2.31.0",
        "numpy>=1.24.3",
        "pandas>=2.1.4",
    ],
    extras_require={
        "dev": [
            "hypothesis>=6.92.1",
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.27.1",
        ]
    },
)
