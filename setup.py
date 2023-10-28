from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1.0",
    description=(
        "Deep Learning training pipeline template"
        "based on pytorch_lightning and hydra"
    ),
    author="Yann Nobrega",
    author_email="yann.nicholas@gmail.com",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(exclude=["tests"]),
)
