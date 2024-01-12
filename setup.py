"""
setup.py
========
"""


from setuptools import setup, find_packages


setup(
    name="machine-learning",
    version="0.0.1",
    author="Frank Lehner",
    author_email="frank-lehner71@t-online.de",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "Cython",
        "scikit-learn",
        "tensorflow",
        "theano",
        "keras",
        "h5py",
        "tables",
        "Pillow",
        "opencv-python",
        "gensim",
        "click",
        "pyprind",
        "nltk",
    ]
)
