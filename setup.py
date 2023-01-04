from setuptools import find_packages, setup
from codecs import open
import os

curfile = os.path.abspath(os.path.dirname(__file__))

PNAME, PTITLE = "multiscaleio", "multiscaleio"
vpath = os.path.join(curfile, PNAME, "__version__.py")

with open("README.md", "r", "utf-8") as rm:
    readme = rm.read()

def get_requirements(path, version: bool = False):
    with open(path) as req:
        requirements = []
        for line in req:
            requirements.append(line.replace("\n", ""))

    if version:
        requirements = {
            info.split(" = ")[0]: info.split(" = ")[1].replace('"', '') 
            for info in requirements
        }
    return requirements

infos = get_requirements(vpath, version=True)

if __name__ == "__main__":
    setup(
        name=infos["__title__"],
        description=infos["__description__"],
        version=infos["__version__"],
        url=infos["__url__"],
        author=infos["__author__"],
        license="Apache 2.0",
        long_description=readme,
        long_description_content_type="text/markdown",
        author_email=infos["__author_email__"],
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python 3.9"
        ],
        packages=[f"{PNAME}.{p}" for p in find_packages(PNAME)],
        package_dir={PTITLE: PNAME},
        py_modules=["settings"],
        include_package_data=True,
        package_data={},
        install_requires=get_requirements("requirements.txt"),
        python_requires=">=3"
    )