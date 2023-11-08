from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]: # in short our function will return a list
    '''
    this functions return the list of requirements !!
    '''
    requirements = []
    with open(file_path) as file:
        requirements=file.readlines() # readlines adds a \n that we dont need
        requirements = [req.replace('\n','') for req in requirements  ] # replacing the \n by blank string


        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Devang',
    author_email='devangchavan0204@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)