# simple setup.py for the frankenstein package

import setuptools

setuptools.setup(
    name="frankenstein",
    version="0.0.1",
    author='Eric Purdy',
    install_requires=[
        'matplotlib',
        'numpy',
        'torch',
        'tqdm',
        'transformers',
        'tokenizers',
        'sklearn',
    ]
)