import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyALE",  # Replace with your own username
    version="0.0.1",
    author="Dana Jomar",
    author_email="dana.jomar@outlook.com",
    description="ALE plots with python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanaJomar/PyALE",
    # packages=setuptools.find_packages(),
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
