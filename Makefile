# Makefile
.PHONY: help install install-dev setup clean test lint format docker-build docker-up docker-down

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev,docs,deployment]"
	pre-commit install

setup: ## Complete project setup
	@echo "Setting up GeoSales Intelligence Platform..."
	cp .env.example .env
	docker-compose up -d postgres redis
	sleep 10
	python scripts/setup/setup_database.py
	python scripts/data/import_sample_data.py
	@echo "Setup complete!"

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

test: ## Run tests
	pytest tests/ -v --cov=backend --cov=ml_engine --cov=stream_processing

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	pytest tests/e2e/ -v

lint: ## Run linting
	flake8 backend/ ml_engine/ stream_processing/
	mypy backend/ ml_engine/ stream_processing/

format: ## Format code
	black backend/ ml_engine/ stream_processing/
	isort backend/ ml_engine/ stream_processing/

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all services
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## View logs
	docker-compose logs -f

train-models: ## Train ML models
	python scripts/ml/train_models.py

deploy-staging: ## Deploy to staging
	./scripts/deployment/deploy.sh staging

deploy-production: ## Deploy to production
	./scripts/deployment/deploy.sh production

backup-db: ## Backup database
	python scripts/data/backup_database.py

restore-db: ## Restore database
	python scripts/data/restore_database.py

generate-docs: ## Generate documentation
	mkdocs build

serve-docs: ## Serve documentation locally
	mkdocs serve

performance-test: ## Run performance tests
	locust -f tests/load_testing/locustfile.py

security-scan: ## Run security scan
	bandit -r backend/ ml_engine/ stream_processing/

check-deps: ## Check for dependency vulnerabilities
	safety check

update-deps: ## Update dependencies
	pip-review --local --interactive

