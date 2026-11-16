# ABOUTME: Package metadata for the icrl-safety pipeline; the code is imported as src.* from the repo root.
# ABOUTME: Install for development with: pip install -e .
from setuptools import setup, find_packages

setup(
    name="icrl-safety",
    version="0.1.0",
    packages=find_packages(
        include=["src", "src.*"],
    ),
    python_requires=">=3.11",
)
