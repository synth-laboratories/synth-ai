from setuptools import find_packages, setup

setup(
    name="zyk",
    version="0.2.34",
    packages=find_packages(),
    install_requires=[
        "openai",
        "pydantic>=2.9.2",
        "diskcache",
        "backoff>=2.2.1",
        "anthropic>=0.34.2",
        "google>=3.0.0",
        "google-generativeai>=0.8.1",
        "together>=1.2.12",
    ],
    author="Josh Purtell",
    author_email="jmvpurtell@gmail.com",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JoshuaPurtell/jazyk",
    license="MIT",
)