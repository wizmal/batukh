import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="batukh",
    version="0.1.0",
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
)
