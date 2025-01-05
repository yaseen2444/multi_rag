from setuptools import find_packages,setup

def get_requirements(file_path):
    '''
    this function returns the requirements and dependencies
    :param file_path:
    :return:
    '''

    requirements=[]

    with open(file_path,"r") as fb:
        requirements=fb.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]
        if "-e." in requirements:
            requirements.remove("-e.")
    return requirements
setup(
    name="rag_builder",
    version="0.0.1",
    author="rishi",
    author_email="mrishe6@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')

)