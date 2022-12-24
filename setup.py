from setuptools import find_packages, setup


setup(name='gato',
      version='0.0.1',
      description='Unofficial Gato: A Generalist Agent',
      url='https://github.com/OrigamiDream/gato.git',
      author='OrigamiDream',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      install_requires=[
          'tensorflow>=2.11',
          'numpy'
      ])
