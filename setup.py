from setuptools import setup, find_packages

setup(
    name="sonic-ml",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'sonic=sonic_ml.main:main',
        ],
    },
    install_requires=[
        'torch',
        'sentencepiece',
        'datasets',
        'tqdm',
        'numpy',
        'transformers',
        'tiktoken',
        'flytekit',
    ],
    author="Rashik Shahjahan",
    author_email="rashikshahjahan@protonmail.com",
    description="A CLI tool for training and evaluating language models",
    url="https://github.com/RashikShahjahan/llm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 