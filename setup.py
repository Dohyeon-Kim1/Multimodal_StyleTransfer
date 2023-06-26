from setuptools import setup

setup(
    name="OBSTwT",
    version="0.1.0",
    description="Package for Object and Bacground Style Transfer with Text.",
    url="https://github.com/Dohyeon-Kim1/Object-Background_StyleTransfer_withText.git",
    install_requires=[
        "diffusers",
        "transforms",
        "scipy",
        "ftfy",
        "accelerate"
        ]
    )
