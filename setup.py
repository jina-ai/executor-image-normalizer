from setuptools import find_packages
import setuptools


setuptools.setup(
    name="crafter-image-normalizer",
    version="2.0",
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description="Executor that reads and normalizes images",
    url="https://github.com/jina-ai/crafter-image-normalizer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where='.', include=['jinahub.*']),
    install_requires=open("jinahub/image_normalizer/requirements.txt").readlines(),
    python_requires=">=3.7"
)