from setuptools import setup, find_packages

# Load requirements
with open("requirements.txt", "r") as req_file:
    requirements = req_file.readlines()

setup(
    name="my-slic",
    version="0.1.0",
    author="Your Name",
    author_email="your-email@example.com",
    description="A brief description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/yourusername/your-repo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
