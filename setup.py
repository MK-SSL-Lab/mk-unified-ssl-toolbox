import os
from pathlib import Path

import setuptools

_PATH_ROOT = Path(os.path.dirname(__file__))


def load_version() -> str:
    version_filepath = _PATH_ROOT / "MK_SSL" / "__init__.py"
    with version_filepath.open() as file:
        for line in file.readlines():
            if line.startswith("__version__"):
                version = line.split("=")[-1].strip().strip('"')
                return version
    raise RuntimeError("Unable to find version string in '{version_filepath}'.")


if __name__ == "__main__":
    name = "MK_SSL"
    version = load_version()
    author = "Kianoosh Vadaei & Melika Shirian"
    author_email = "kia.vadaei@gmail.com"
    description = "A Self-Supervised Learning Library"
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    python_requires = ">=3.10"

    install_requires = [
        "axial_positional_embedding",
        "colorlog",
        "editdistance",
        "einops",
        "huggingface_hub",
        "jiwer",
        "joblib",
        "numpy",
        "opencv_python",
        "optuna",
        "pandas",
        "peft",
        "Pillow",
        "plotly",
        "scikit_learn",
        "setuptools",
        "torch",
        "torch_geometric",
        "torchaudio",
        "torcheval",
        "torchmetrics",
        "torchvision",
        "tqdm",
        "transformers",
        "wandb",
    ]

    packages = setuptools.find_packages()

    project_urls = {
        "Github": "https://github.com/MK-SSL-Lab/mk-unified-ssl-toolbox",
    }

    classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    ]

    setuptools.setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        description=description,
        license="MIT",
        license_files=["LICENSE"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=install_requires,
        python_requires=python_requires,
        packages=packages,
        classifiers=classifiers,
        project_urls=project_urls,
    )
