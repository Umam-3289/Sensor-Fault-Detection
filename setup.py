from setuptools import find_packages,setup
from typing import List

#just update
#DockerFIle --> Dockerfile commit changes in Github --> pushed into local system

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    return requirements

setup(
    name="Sensor Fault Detection",
    version="0.0.1",
    author="Umam Khan",
    author_email="umamkhan9931@gmail.com",
    install_requirements=get_requirements("requirements.txt"),
    packages=find_packages()
)