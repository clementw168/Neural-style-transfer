PYTHON_FILE_PATHS = `(find . -iname "*.py" -not -path "**/.venv/*")`

lab: ## Launch Jupyterlab instance
	poetry run jupyter-lab

lab-install: ## Create jupyter-lab kernel
	poetry run ipython kernel install --name "neural-style-transfer" --user

lab-remove: ## Remove jupyter-lab kernel
	poetry run jupyter kernelspec uninstall neural-style-transfer

install: ## Install element dependencies
	poetry install

install-hard: ## Clear and install element dependencies
	rm -rf poetry.lock .venv && poetry install

poetry-upgrade: ## Upgrade poetry and dependencies
	poetry self update
	poetry run pip install --upgrade pip wheel setuptools
	poetry update

poetry-sort: ## Sort poetry dependencies alphabetically
	poetry run toml-sort pyproject.toml --all --in-place

pylint: ## Run Pylint
	poetry run pylint -s no --rcfile=.pylintrc $(PYTHON_FILE_PATHS)

isort: ## Run Isort
	poetry run isort --check-only $(PYTHON_FILE_PATHS)

isort-fix: ## Run Isort with automated fix
	poetry run isort $(PYTHON_FILE_PATHS)

black: ## Run Black
	poetry run black --check --quiet $(PYTHON_FILE_PATHS)

black-fix: ## Run Black with automated fix
	poetry run black $(PYTHON_FILE_PATHS)

help: ## Description of the Makefile commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

