from setuptools import setup, find_packages

# find packages
packages = find_packages(exclude=("tests", "tests.*"))

# setup
setup(
    name='pytorch-minimize',
    description='Newton and Quasi-Newton optimization with PyTorch',
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