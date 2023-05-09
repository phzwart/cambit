from setuptools import setup, find_packages

setup(
    name='latexp',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'dash',
        'pandas',
        'plotly'
    ],
    entry_points={
        'console_scripts': [
            'latexp=latexp:main'
        ]
    },
    include_package_data=True,
    zip_safe=False
)
