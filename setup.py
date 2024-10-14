import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openmatch-thunlp",
    version="0.0.1",
    author="Shi Yu",
    author_email="yushi17@foxmail.com",
    description="An python package for research on Information Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Text Processing :: Indexing",
        "Intended Audience :: Information Technology",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "transformers>=4.21.3",
        "datasets>=2.10.1",
        "sentencepiece",
    ],
)
