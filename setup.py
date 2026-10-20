from setuptools import setup, find_packages
import os
import sys

# Base dependencies required for all installations
base_requires = [
    "IPython",
    "matplotlib",
    "gym",
    "gym_sokoban",
    "peft",
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    # "flash-attn==2.7.4.post1",
    "hydra-core",
    "numpy",
    "pandas",
    "pybind11",
    "ray>=2.10",
    "tensordict>=0.8.0,<0.9.0",
    "transformers",
    "vllm==0.8.2",
    "wandb",
    "gymnasium",
    "gymnasium[toy-text]",
    "pyarrow>=15.0.0",
    "pylatexenc",
    "torchdata",
    "debugpy",
    "together",
    "anthropic",
    "faiss-cpu==1.11.0",
]

# Optional dependencies for webshop environment
webshop_requires = [
    "beautifulsoup4",
    "cleantext",
    "flask",
    "html2text",
    "rank_bm25",
    "pyserini",
    "thefuzz",
    "gdown",
    "spacy",
    "rich",
]

# Optional dependencies for lean environment
lean_requires = [
    "kimina-client",
]

# Optional dependencies for search environment
# Note: the retrieval server (scripts/retrieval/server.py) additionally needs:
#   flask, faiss-cpu, sentence-transformers
search_requires = [
    "requests",
    "mcp>=1.26,<2",
]

setup(
    name='ragen',
    version='0.1',
    package_dir={'': '.'},
    packages=find_packages(include=['ragen']),
    author='RAGEN Team',
    author_email='',
    acknowledgements='',
    description='',
    install_requires=base_requires,
    extras_require={
        "webshop": webshop_requires,
        "lean": lean_requires,
        "search": search_requires,
        "all": webshop_requires + lean_requires + search_requires,
    },
    package_data={'ragen': ['*/*.md']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
    ]
)
