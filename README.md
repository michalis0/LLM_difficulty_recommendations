# LLM_difficulty_recommendations

## Description

This project contains the code used for our research work described in the article [**Large Language Models for Difficulty Estimation of Foreign Language Content with Application to Language Learning**](https://arxiv.org/abs/2309.05142). It contains all the code for our experiments as well as the notebooks used to generate the figures in the article. The aim of the project is to enable :
- The classification of French texts according to their difficulty,
- The classification of French texts according to their subject,
- The recommendation of French texts according to the two previous criteria.
Each of these objectives is dealt with and detailed in a separate file.

## Installation 🐼

### Requirements 🐨

This project uses the following tools:
- [Pyenv](https://github.com/pyenv/pyenv-installer) : Python version manager
- [Poetry](https://python-poetry.org/docs/#installation) : Python package manager
Tools that you can install using the following commands:
```bash
> curl https://pyenv.run | bash
> curl -sSL https://install.python-poetry.org | python3 -
```

Certains notebooks nécessitent également les installations suivantes :
- MySQL : [Installation](https://dev.mysql.com/doc/mysql-installation-excerpt/5.7/en/)
- Un nom d'utilisateur et un mot de passe pour se connecter au cluster [CURNAGL](https://wiki.unil.ch/ci/books/high-performance-computing-hpc/page/curnagl) de l'université de Lausanne utilisé pour l'entrainement des modèles les plus lourds. Ce code peut cependant être exécuté sur une machine locale sans utiliser le package [**Slurmray**](https://github.com/hjamet/SLURM_RAY).

### With Pyenv & Poetry 🐻

Once you have installed the above tools, you can install the project using the following commands:
```bash
> pyenv install 3.9.13 # Installs the version of Python used in the project
> pyenv local 3.9.13 # Defines the version of Python used in the project
> poetry install # Installs project dependencies
```

### Without Pyenv & Poetry 🐙

If you don't want to use Pyenv and Poetry, you can install the project dependencies using the following command:
```bash
> python -m venv .venv # Creates a virtual environment
> source .venv/bin/activate # Activates the virtual environment
> pip install -r requirements.txt # Installs project dependencies
```

## Repository Convention & Architecture 🦥

### Architecture 🦜