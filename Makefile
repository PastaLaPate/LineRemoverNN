.DEFAULT_GOAL := help
SHELL         := /bin/bash
CYAN  := \033[0;36m
GREEN := \033[0;32m
NC    := \033[0m

.PHONY: help
help: ## Show this help message.
	@echo -e ""
	@echo -e "  $(CYAN)Lineremovernn$(NC)"
	@echo -e ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ \
	        { printf "  $(GREEN)%-22s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo -e ""

.PHONY: install
install: ## Installs all python deps.
	bash ./dev-install.sh && \
	uv sync --locked --all-extras --dev

.PHONY: pull
update: ## Just runs git pull.
	git pull

.PHONY: dev
dev: ## Run main module
	uv run lineremovernn

.PHONY: build
build: ## Build CPP lib.
	bash ./dev-install.sh

.PHONY: build-run
build-run: ## Builds CPP lib and runs module.
	bash ./dev-install.sh && uv run lineremovernn

.PHONY: hooks
hooks: ## Install pre-commit hooks.
	uv run pre-commit install