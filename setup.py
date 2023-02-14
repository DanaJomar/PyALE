import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyALE",
    version="1.1.3",
    author="Dana Jomar",
    author_email="dana.jomar@outlook.com",
    description="ALE plots with python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanaJomar/PyALE",
    packages=setuptools.find_packages(exclude=["*.tests"]),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "statsmodels",
    ],
    extras_require={
        "dev": ["coverage"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=["ALEPlots ALE feature effect interpretable ML"],
)
