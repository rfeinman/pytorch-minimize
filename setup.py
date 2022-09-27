from setuptools import setup, find_packages
from pathlib import Path

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# find packages
packages = find_packages(exclude=("tests", "tests.*"))

# setup
setup(
    name='pytorch-minimize',
    version='0.0.1',
    description='Newton and Quasi-Newton optimization with PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://pytorch-minimize.readthedocs.io',
    author='Reuben Feinman',
    author_email='reuben.feinman@nyu.edu',
    license='MIT Licence',
    packages=packages,
    zip_safe=False,
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.6',
        'torch>=1.9.0'
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)