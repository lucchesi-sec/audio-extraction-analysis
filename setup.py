from setuptools import find_packages, setup

setup(
    name="audio-extraction-analysis",
    version="1.0.0+emergency",
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"src": "src"},
    install_requires=[line for line in open("requirements-core.txt").read().splitlines() if line],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "audio-extraction=src.cli:main",
            "audio-extraction-analysis=src.cli:main",
        ],
    },
)
