# creating ML application as package

from setuptools import find_packages,setup
from typing import  List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name = 'Job_Cls',
    version = '0.0.1',
    author = 'Phat',
    author_email = 'dophat694@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt'),
)