from setuptools import find_packages, setup

setup(
    name="bino-utils",
    version="0.0.1",
    author="Semmelhack Lab",
    author_email="kclamar@connect.ust.hk",
    description="Utility functions for binocular paper",
    packages=find_packages(),
    package_data={},
    include_package_data=True,
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=[],
    extras_require={},
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
