from setuptools import setup, find_packages

setup(name='labse_tools',
      version="0.1",
      description="Test for pip install git+",
      url="https://github.com/Ernst-Y0ung/LaBSE",
      install_requires=['numpy', 'tensorflow>=2.0', 'tensorflow_hub', 'bert-for-tf2'],
      packages=find_packages(),
      )
