from setuptools import setup, find_packages

setup(
    name="turboquant",
    version="0.1.0",
    description="TurboQuant: Near-optimal KV cache quantization for LLM inference",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Implementation based on Zandieh et al. (ICLR 2026)",
    url="https://github.com/0xSero/turboquant",
    packages=find_packages(),
    package_data={"turboquant": ["codebooks/*.json"]},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "numpy",
        "scipy",
    ],
    extras_require={
        "vllm": ["vllm>=0.16"],
        "triton": ["triton>=3.0"],
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "turboquant=turboquant.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
