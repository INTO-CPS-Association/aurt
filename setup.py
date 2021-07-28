import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aurt",
    version="0.0.1",
    author="Emil Madsen, Daniella Tola, Claudio Gomes",
    author_email="ema@ece.au.dk",
    description="A robot dynamic parameters calibration toolbox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.au.dk/software-engineering/aurt",
    packages=["aurt"],
    install_requires=[
        "numpy>=1",
        "sympy>=1",
        "pandas>=1",
        "scikit-learn>=0.24",
        "matplotlib>=1"
    ],
    # extras_require={
    #     "dev": ["matplotlib"]
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={"console_scripts": ["aurt=aurt.cli:main"]},
)
