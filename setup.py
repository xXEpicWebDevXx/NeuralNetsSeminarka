from setuptools import find_packages, setup

setup(
    name='neura',
    packages=find_packages(),
    version='0.1.0',
    description='Simple neural networks library, made using only numpy',
    install_requires=["numpy"],
    author='Lukáš Novák',
)
