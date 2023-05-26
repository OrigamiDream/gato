from setuptools import find_packages, setup

setup(
    name='gato-tf',
    version='0.0.3',
    description='Unofficial Gato: A Generalist Agent',
    url='https://github.com/OrigamiDream/gato.git',
    author='OrigamiDream',
    author_email='sdy36071@naver.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.11',
        'flowchain>=0.0.4'
    ],
    keywords=[
        'deep learning',
        'gato',
        'tensorflow',
        'generalist agent'
    ]
)
