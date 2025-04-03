import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="baremetal-nn",
    version="2025.04.02",
    author="-T.K.-",
    author_email="t_k_233@outlook.com",
    description="Convert PyTorch models to Baremetal C/C++ code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ucb-bar/Baremetal-NN",
    project_urls={
        "API Documentation": "https://ucb-bar.github.io/Baremetal-NN/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "torch",
        "torchtune",
        "torchao",
        "tabulate",
    ],
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
)