from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gym_simpletetris",
    version="0.5.0",
    author="GIJaws",
    license="MIT",
    description="A simple Tetris engine for OpenAI Gym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GIJaws/gym-simpletetris",
    project_urls={
        "Bug Tracker": "https://github.com/GIJaws/gym-simpletetris/issues",
    },
    install_requires=[
        "numpy==1.26.4",
        "pygame>=2.6.0",
        "gymnasium>=0.29.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "tetris-human-play=scripts.run_human_play:main",
        ],
    },
    python_requires=">=3.10",
)
