from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("dependency_links.txt") as f:
    dependency_links = f.read().splitlines()

setup(
    name="umbertobot",
    version="0.3",
    description="Utils for document processing using language models",
    url="https://github.com/lambdaofgod/umbertobot",
    author="Jakub Bartczuk",
    packages=find_packages(),
    install_requires=requirements,
    dependency_links=dependency_links,
)
