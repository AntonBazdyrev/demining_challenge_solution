# Quick Setup Guide

## 0. Env
1. [Install Poetry](https://python-poetry.org/docs/#installation)
  - The easiest way is to use [Official Installer guide](https://python-poetry.org/docs/#installing-with-the-official-installer).
  - Pay attention to `poetry --version`. in order to install correct version do : `curl -sSL https://install.python-poetry.org | python3 - --version 1.8.3`. If you have already installed another version - simply change it with next command: `poetry self update 1.8.3`
2. Configure Poetry to create Env in local folder - `poetry config virtualenvs.in-project true`
3. `poetry shell`
  - Check `poetry env info` you need to check that you are using `Python:     3.11.*`. Otherwise do `poetry env use 3.11`. You need to do it only once, when you create your first env.
5. In order to deactivate env - `exit`
6. In order to remove env - `rm .venv -rf`

## 1. Git pull mmdetection

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```

## 2. Train run

```bash
python mmdetection/tools/train.py rtmdet_tiny_1xb4_20e.py
```