import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "numpy",
    "tensorboard>=2.0.2",
    "tqdm",
]

torch_requirements = [
    "torch>=1.5.1",
    "torchvision>=0.6.1",
    "Pillow>=6.2.0",
]

tf_requirements = [
    "tensorflow>=2.3.0",
    "tensorflow-addons>=0.11.2",
]

setuptools.setup(
    name="batukh",
    version="0.1.1",
    author="Murtaza, Wajid, Naveed",
    author_email="batukhorg@gmail.com",
    description="Document recognizer for multiple languages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KoshurNizam/batukh",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    extras_require={
        "tf": tf_requirements,
        "torch": torch_requirements,
        "full": tf_requirements+torch_requirements
    }
)
