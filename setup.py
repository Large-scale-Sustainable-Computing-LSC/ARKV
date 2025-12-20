from __future__ import annotations

from setuptools import find_namespace_packages, setup


setup(
    name="akcb",
    version="0.1.0",
    description="AKCB: Adaptive and Resource-Efficient KV Cache Management",
    packages=find_namespace_packages(include=["akcb*"]),
    include_package_data=True,
    python_requires=">=3.10",
)
