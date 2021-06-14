import setuptools


setuptools.setup(
    name="executor-image-normalizer",
    version="2.0",
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description="Executor that normalizes images",
    url="https://github.com/jina-ai/executor-image-normalizer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    py_modules=['jinahub.image.normalizer'],
    package_dir={'jinahub.image': '.'},
    install_requires=open("requirements.txt").readlines(),
    python_requires=">=3.7",
)
