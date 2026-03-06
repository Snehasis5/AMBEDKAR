"""
Setup configuration for the AMBEDKAR package.

Install in development mode:
    pip install -e .

Install with optional extras:
    pip install -e ".[dev]"      # testing + linting
    pip install -e ".[notebook]" # Jupyter notebook support
"""

from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="ambedkar",
    version="1.0.0",
    description=(
        "AMBEDKAR: Training-free, inference-time debiasing via "
        "fairness-aware speculative decoding (ACL 2026)"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/ambedkar-acl2026/AMBEDKAR",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
        ],
        "wordnet": [
            "nltk>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ambedkar-evaluate=scripts.run_evaluation:main",
            "ambedkar-train-verifier=scripts.train_verifier:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "fairness", "bias-mitigation", "speculative-decoding",
        "large-language-models", "india", "caste", "religion",
        "constitutional-ai", "debiasing", "nlp",
    ],
)
