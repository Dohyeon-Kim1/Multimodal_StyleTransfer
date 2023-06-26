from setuptools import find_packages, setup

setup(
    name="OBSTwT",
    version="0.1.0",
    description="Package for Object and Bacground Style Transfer with Text.",
    packages=find_packages(exclude="notebooks"),
    install_requires=[
        "diffusers",
        "transforms",
        "scipy",
        "ftfy",
        "accelerate"
        ]
    )
