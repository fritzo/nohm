import sys

from setuptools import find_packages, setup

# READ README.md for long description on PyPi.
# This requires uploading via twine, e.g.:
# $ python setup.py sdist bdist_wheel
# $ twine upload --repository-url https://test.pypi.org/legacy/ dist/*  # test version
# $ twine upload dist/*
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to convert README.md to rst:\n  {}\n".format(e))
    sys.stderr.flush()
    long_description = ""


def remove_badges(text):
    """
    Remove badges since they will always be obsolete.
    """
    lines = text.split("\n")
    lines.reverse()
    while lines and lines[-1]:
        lines.pop()
    while lines and not lines[-1]:
        lines.pop()
    lines.reverse()
    return "\n".join(lines)


long_description = remove_badges(long_description)

setup(
    name="nohm",
    version="0.0.0",
    description="Nondeterministic Optimal Higher-order Machine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["nohm", "nohm.*"]),
    package_data={"nohm": ["py.typed"]},
    entry_points={
        "console_scripts": [
            "nohm.benchmark=nohm.benchmark:main",
        ],
    },
    url="https://github.com/fritzo/nohm",
    author="Fritz Obermeyer",
    python_requires=">=3.7",
    install_requires=[],
    extras_require={
        "test": [
            "black",
            "flake8",
            "isort>=5.0",
            "mypy",
            "pytest",
            "pytest-xdist",
        ],
    },
    keywords="functional programming language interaction nets",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
