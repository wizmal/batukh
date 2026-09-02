import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


tensorflow_dependencies = [
    "tensorflow>=2.21.0,<2.22",
    "tensorboard>=2.21.0,<2.22",
    "opencv-python>=4.10.0",
]

setuptools.setup(
    name="batukh",
    version="0.1.0",
    author="Murtaza, Wajid, Naveed",
    author_email="batukhorg@gmail.com",
    description="Document recognizer for multiple languages.",
    license="GPL-3.0-only",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wizmal/batukh",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "tensorflow": tensorflow_dependencies,
        "torch": ["torch>=1.5.1", "torchvision>=0.6.1"],
        "all": tensorflow_dependencies + ["torch>=1.5.1", "torchvision>=0.6.1"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
