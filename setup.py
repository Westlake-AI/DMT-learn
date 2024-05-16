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
 

setup(
    name='DMTEV-learn',
    version='0.0.1',
    description='An Explainable Deep Network for Dimension Reduction (EVNet)',
    long_description=README,
    long_description_content_type='text/markdown',
    author='zangzelin',
    author_email='zangzelin@westlake.edu.cn',
    python_requires=">=3.7",
    requires=parse_requirements(PARENT / 'requirements.txt'),
    packages=find_packages(),  # 系统自动从当前目录开始找包
    exclude_package_data=[
        'data/',
        '__pycache__/',
        '.vscode/',
        'lightning_logs/',
        'wandb/',
        'save_near_index/',
        'baseline/',
        'Embedding/',
        'save_html/',
        'save_checkpoint/',
        'save_checkpoint_use/',
        'tensorflow/*',
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA"
        "Programming Language :: Python::3.7",
        "Programming Language :: Python::3.8",
        "Programming Language :: Python::3.9",
        "Programming Language :: Python::3.10",
        "Programming Language :: Python::3.11",
        "Programming Language :: Python::3.12",
    ],
)