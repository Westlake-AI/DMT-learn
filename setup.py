from pathlib import Path

from setuptools import setup
from setuptools import find_packages


# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')


def get_version():
    version_file = f'{PARENT}/version.py'
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']   
 

def parse_requirements(file_path: Path):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.
    Args:
        file_path (str | Path): Path to the requirements.txt file.
    Returns:
        (List[str]): List of parsed requirements.
    """
 
    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line.split('#')[0].strip())  # ignore inline comments

    return requirements


setup(
    name='dmtev-learn',
    version=get_version(),
    description='An Explainable Deep Network for Dimension Reduction (EVNet)',
    long_description=README,
    long_description_content_type='text/markdown',
    author='zangzelin',
    author_email='zangzelin@westlake.edu.cn',
    packages=find_packages(),  # 系统自动从当前目录开始找包
    python_requires=">=3.7",
    install_requires=parse_requirements(PARENT / 'requirements.txt'),
    dependency_links=['https://download.pytorch.org/whl/cu118/torch', 'https://download.pytorch.org/whl/cu118/torchvision', 'https://download.pytorch.org/whl/cu118/torchaudio'],
    exclude_package_data={
        '__pycache__': ['*'],
    },
    extras_require={
        "eval": [
            "pandas",
            "plotly",
            "matplotlib",
            "umap-learn",
            "scanpy"
        ]
    },
    platforms=['linux_x86_64'],
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
    keywords="dimension reduction manifold deep network",
)