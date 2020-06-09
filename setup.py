import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vppy",
    version="0.0.1",
    author="Erwin van Duijnhoven",
    author_email="erwin.duijnhoven@gmail.com",
    description="A package that calculates vanishing points",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'opencv-python',
        'numpy',
        'sklearn'
    ],
    entry_points={
        "console_scripts": ["vppy=vppy.__main__:main"]
    },
)