from setuptools import setup, find_packages

setup(
    name="latexp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "plotly",
        "jupyter-dash", 
        "dash",
        "numpy",
        "pandas"
    ],
    python_requires=">=3.7",
    author="Your Name",
    description="Interactive latent space explorer for image data",
    keywords="visualization, machine learning, latent space, dash",
) 