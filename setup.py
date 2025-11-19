from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="akcb",
    version="0.1.0",
    author="Jianlong Lei",
    description="Adaptive KV Caches under Budget for Efficient Long-Context LLM Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JianlongLei/AKCB",
    packages=find_packages(exclude=["experiments*", "tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "tqdm",
    ],
    extras_require={
        "eval": [
            "lm-eval>=0.4.0",
            "rouge>=1.0.1",
            "jieba>=0.42.1",
            "fuzzywuzzy>=0.18.0",
            "python-Levenshtein>=0.21.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
)
