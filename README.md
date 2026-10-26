# Healthcare AI Agent with Tool-Calling

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-FF6B6B)](https://www.langchain.com/langgraph)
[![OpenAI](https://img.shields.io/badge/OpenAI%20Compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0%2B-D71F00?logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![DeepEval](https://img.shields.io/badge/DeepEval-LLM%20Evaluation-6366F1)](https://www.deepeval.com/)
[![pytest](https://img.shields.io/badge/pytest-Testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, modular AI Agent built with Python, LangChain, LangGraph, and SQLAlchemy that connects to a PostgreSQL database containing clean medical records. The agent leverages OpenAI-compatible tool-calling to intelligently query, analyze, and retrieve patient information through natural language interactions.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Database Schema](#database-schema)
- [Available Tools](#available-tools)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Error Handling](#error-handling)
- [Security Considerations](#security-considerations)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Overview

This agent is designed for healthcare data exploration with a focus on clean architecture, type safety, and production-readiness. It provides three specialized tools to query a PostgreSQL database of 55,500 pre-processed medical records across six common conditions: **Arthritis**, **Diabetes**, **Hypertension**, **Obesity**, **Cancer**, and **Asthma**.

## Key Features

- **Clean Architecture**: Strict separation of concerns across configuration, database, repository, tools, and agent orchestration layers
- **Tool-Calling Capabilities**: Three purpose-built tools for patient search, billing summaries, and condition-based queries
- **ReAct Agent Pattern**: Implements LangGraph's `create_react_agent` for reasoning and acting with tool use
- **OpenAI-Compatible LLM**: Configured for Unsloth endpoint with Gemma 3 4B IT GGUF model
- **Robust Error Handling**: Comprehensive validation, database error recovery, and graceful failure handling
- **Type Safety**: Full type hints throughout with Pydantic schemas for tool inputs
- **Extensive Documentation**: Detailed docstrings, inline comments, and execution flow explanations
- **Interactive CLI**: User-friendly command-line interface with verbose tracing and helpful commands
- **Modern Tooling**: Managed with [uv](https://github.com/astral-sh/uv) for lightning-fast dependency resolution
- **Secure by Design**: Parameterized queries, environment-based secrets, connection pooling with health checks

## Architecture

The project follows Clean Architecture principles with unidirectional dependencies, ensuring testability and maintainability:

```mermaid
graph TD
    A[main.py - CLI Presentation] --> B[agent.py - ReAct Agent Orchestration]
    B --> C[tools.py - LangChain Tool Definitions]
    C --> D[repository.py - Repository Pattern Data Access]
    D --> E[database.py - SQLAlchemy Engine & Sessions]
    E --> F[config.py - Configuration & Constants]
    
    style A fill:#4A90E2,stroke:#2E5C8A,color:#fff
    style B fill:#FF6B6B,stroke:#C44545,color:#fff
    style C fill:#F39C12,stroke:#D68910,color:#fff
    style D fill:#27AE60,stroke:#1E8449,color:#fff
    style E fill:#8E44AD,stroke:#6C3483,color:#fff
    style F fill:#34495E,stroke:#2C3E50,color:#fff
```

## Technology Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| [**Python**](https://www.python.org/) | Core language | 3.9+ |
| [**uv**](https://github.com/astral-sh/uv) | Fast Python package & project manager | Latest |
| [**LangChain**](https://www.langchain.com/) | LLM framework and tool abstractions | >=0.3.0 |
| [**LangGraph**](https://www.langchain.com/langgraph) | Agent orchestration with ReAct pattern | >=0.2.0 |
| [**LangChain OpenAI**](https://python.langchain.com/docs/integrations/llms/openai) | OpenAI-compatible client integration | >=0.2.0 |
| [**OpenAI**](https://platform.openai.com/docs/api-reference) | API client library | >=1.40.0 |
| [**SQLAlchemy**](https://www.sqlalchemy.org/) | ORM and database toolkit | >=2.0.32 |
| [**psycopg2-binary**](https://www.psycopg.org/) | PostgreSQL adapter | >=2.9.9 |
| [**Pydantic**](https://docs.pydantic.dev/) | Data validation and settings management | >=2.8.0 |
| [**python-dotenv**](https://pypi.org/project/python-dotenv/) | Environment variable management | >=1.0.1 |

## Database Schema

The agent connects to PostgreSQL with the `healthcare_records` table containing **55,500 pre-processed records** spanning 6 key medical conditions.

| Column | Type | Description |
|--------|------|-------------|
| `id` | `INT8` / `SERIAL` PRIMARY KEY | Unique patient record identifier |
| `name` | `TEXT` | Patient first name |
| `surname` | `TEXT` | Patient last name |
| `age` | `INT4` | Patient age |
| `gender` | `TEXT` | Patient gender |
| `blood_type` | `TEXT` | Blood type (e.g., A+, O-, AB+) |
| `medical_condition` | `TEXT` | Primary medical condition |
| `date_of_admission` | `DATE` | Hospital admission date |
| `doctor` | `TEXT` | Attending physician name |
| `hospital` | `TEXT` | Hospital/facility name |
| `insurance_provider` | `TEXT` | Insurance provider name |
| `billing_amount` | `NUMERIC(12, 2)` | Total billing amount in currency |
| `room_number` | `INT4` | Assigned room number |
| `admission_type` | `TEXT` | Type of admission (e.g., Emergency, Elective, Urgent) |
| `discharge_date` | `DATE` | Hospital discharge date |
| `medication` | `TEXT` | Prescribed medication |
| `test_results` | `TEXT` | Medical test results |

## Available Tools

### 1. `search_patient_by_name(first_name: str, last_name: str)`

Searches for patients by first and last name using case-insensitive exact matching. Returns complete patient profiles with all available fields.

**Use Cases:**
- Locate a specific patient by name
- Retrieve full medical record details
- Gather information before billing queries

**Returns:** JSON array of matching patient records with complete details.

### 2. `get_patient_billing_summary(patient_id: int)`

Retrieves billing information, hospital details, insurance provider, and automatically calculates length of stay in days from admission and discharge dates.

**Use Cases:**
- Get financial/billing information
- Calculate hospital stay duration
- Retrieve administrative details for a specific patient

**Returns:** Structured billing summary including calculated `length_of_stay_days`.

### 3. `query_patients_by_condition(medical_condition: str, limit: int = 5)`

Filters patients by medical condition and returns a condensed summary view. Supports all six dataset conditions with configurable result limits (1-100).

**Supported Conditions:** `Arthritis`, `Diabetes`, `Hypertension`, `Obesity`, `Cancer`, `Asthma`

**Use Cases:**
- Population-level queries by diagnosis
- Research on specific medical conditions
- Generate condition-based patient lists

**Returns:** Summary objects with key demographics and admission details.

## Prerequisites

- **Python**: Version 3.9 or higher
- **Package Manager**: [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- **Database**: PostgreSQL instance (see [Data Setup & ETL Pipeline](#data-setup--etl-pipeline) below)
- **LLM Endpoint**: OpenAI-compatible endpoint configured through `UNSLOTH_BASE_URL` in `.env`

## Data Setup & ETL Pipeline

This project includes an ETL pipeline to generate and populate the `healthcare_records` table with clean medical data. The dataset contains **55,500 pre-processed records** spanning 6 medical conditions: *Arthritis*, *Diabetes*, *Hypertension*, *Obesity*, *Cancer*, and *Asthma*.

### Option 1: Using Docker Compose (Recommended)

A `docker-compose.yml` file is provided to quickly spin up a PostgreSQL instance with the required configuration.

```bash
# Start PostgreSQL database
docker-compose up -d

# Verify the database is running
docker-compose ps
```

The default configuration will create a database named `healthcare` accessible on `localhost:5434`.

### Option 2: Manual PostgreSQL Setup

```sql
-- Create database and user
CREATE DATABASE healthcare;
CREATE USER healthcare WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE healthcare TO healthcare;
```

### Running the ETL with Jupyter Notebook

The project includes `healthcare_etl.ipynb` which handles data generation, cleaning, transformation, and loading into PostgreSQL.

#### Prerequisites for ETL

```bash
# Install Jupyter and additional dependencies for ETL
uv add jupyter pandas numpy faker
```

#### Execute the ETL Pipeline

1. **Start Jupyter Notebook**
```bash
uv run jupyter notebook healthcare_etl.ipynb
```

2. **Configure Database Connection in Notebook**  
   Open the notebook and ensure the database connection parameters match your setup:
```python
# Example connection string used in notebook
DATABASE_URL = "postgresql://healthcare:password@localhost:5434/healthcare"
```

3. **Run All Cells**  
   Execute the notebook cells in order. The pipeline will:
   - Generate synthetic healthcare data for all 6 medical conditions
   - Clean and validate the data (handle missing values, normalize formats)
   - Create the `healthcare_records` table with the correct schema
   - Insert 55,500 records into PostgreSQL
   - Perform data quality checks and validation

4. **Verify Data Loading**
```sql
-- Connect to your database and verify
SELECT COUNT(*) FROM healthcare_records;
-- Expected: 55500

-- Check distribution by condition
SELECT medical_condition, COUNT(*) 
FROM healthcare_records 
GROUP BY medical_condition 
ORDER BY COUNT(*) DESC;
```

### Raw Data

The `raw_data/` directory contains source data files used by the ETL pipeline for reference.

> **Note:** The notebook uses `faker` and other libraries to generate realistic medical records. Ensure you have sufficient disk space and time for the generation process (may take several minutes).


## Installation

### 1. Clone or navigate to the project

```bash
cd /home/mmagni/Documents/dev/mine/healthcare-agent-poc
```

### 2. Install dependencies with uv

```bash
uv sync
```

This will create a virtual environment, resolve dependencies, and install the package in editable mode.

### 3. Set up environment variables

```bash
cp .env.example .env
```

## Configuration

Edit the `.env` file with your environment-specific values:

```env
# PostgreSQL Database Configuration
DATABASE_URL=postgresql://healthcare:password@localhost:5434/healthcare

# Unsloth OpenAI-Compatible LLM Configuration
UNSLOTH_BASE_URL=<your OpenAI-compatible LLM endpoint>
UNSLOTH_API_KEY=<your Unsloth API key>
UNSLOTH_MODEL=<your Unsloth model identifier>

# Optional: Agent Configuration
AGENT_TEMPERATURE=0.0        # 0.0 = deterministic, higher = more creative
AGENT_MAX_ITERATIONS=10      # Maximum tool-calling/reasoning iterations
AGENT_VERBOSE=true           # Enable detailed execution trace
```

> **Note:** Required connection settings are loaded from environment variables (or the root `.env` file via `python-dotenv`). `config.py` does not provide fallback endpoint or credential values.

## Usage

### Interactive CLI Mode

Start the agent in interactive mode:

```bash
uv run healthcare-agent
```

Or use Python module execution:

```bash
uv run python -m healthcare_agent.main
```

### Example Queries

```text
You: Find patient John Smith
You: Get billing summary for patient ID 12345
You: Show me 5 patients with Diabetes
You: Find Jane Doe and show their billing summary
You: List 10 patients diagnosed with Cancer
You: What are the details for patient with ID 67890?
```

### Special CLI Commands

| Command | Description |
|---------|-------------|
| `help` | Display usage examples and query patterns |
| `tools` | Show detailed information about all available tools |
| `clear` | Clear the terminal screen and redisplay banner |
| `exit`, `quit`, `q` | Exit the interactive session |

### Programmatic Usage

```python
from healthcare_agent.agent import healthcare_agent

# Query the agent programmatically
response = healthcare_agent.query("Find patient John Smith and show billing details")
if response["success"]:
    print(response["answer"])
else:
    print(f"Error: {response['error']}")
```

## Project Structure

```text
healthcare-agent-poc/
├── .env.example                  # Example environment configuration
├── .gitignore                    # Local secrets and generated-file exclusions
├── .python-version               # Python version selected by uv
├── docker-compose.yml            # Local PostgreSQL service on host port 5434
├── healthcare_agent/             # Main application package
│   ├── __init__.py               # Package exports
│   ├── agent.py                  # Agent orchestration and execution logic
│   ├── config.py                 # Environment configuration and constants
│   ├── database.py               # SQLAlchemy engine and session management
│   ├── main.py                   # CLI entry point and interactive loop
│   ├── repository.py             # Data access layer for healthcare_records
│   └── tools.py                  # Tool definitions wrapping repository methods
├── raw_data/
│   └── healthcare_dataset.csv    # Source healthcare records for ETL
├── tests/                        # Automated test suite
│   ├── __init__.py
│   ├── conftest.py               # Shared pytest fixtures
│   ├── test_agent_evaluation.py  # Agent behavior and configuration tests
│   ├── test_repository.py        # Repository tests
│   └── test_tools.py             # Tool tests
├── healthcare_etl.ipynb          # CSV transformation and PostgreSQL load notebook
├── LICENSE                       # MIT License
├── pyproject.toml                # Project metadata and uv dependencies
├── pytest.ini                    # Pytest discovery and marker configuration
├── README.md                     # Project documentation
└── uv.lock                       # Locked dependency versions
```

## Error Handling

The application implements comprehensive error handling across all layers:

| Layer | Error Handling |
|-------|----------------|
| **Database** | Connection validation, transaction rollback, pool health checks, SQLAlchemy error wrapping |
| **Repository** | Input validation, parameterized queries, safe type conversion, None handling |
| **Tools** | Pydantic schema validation, structured JSON error responses, graceful exception catching |
| **Agent** | LLM error handling, recursion limits, empty query validation, database connectivity checks |
| **CLI** | User input validation, KeyboardInterrupt/EOF handling, fatal error reporting |

All errors are returned as structured, user-friendly messages without exposing sensitive system details.

## Security Considerations

- **Secrets Management**: All credentials loaded from environment variables, never hardcoded
- **SQL Injection Prevention**: All queries use SQLAlchemy parameterized statements
- **Connection Security**: Connection pooling with `pool_pre_ping` for stale connection detection
- **API Key Protection**: LLM API keys externalized via environment configuration
- **Data Privacy**: Agent only returns requested information with no cross-record data leakage
- **Input Validation**: All tool inputs validated via Pydantic schemas with constraints
- **No Sensitive Logging**: Database URLs displayed with credentials masked in CLI banner

## Development

### Code Quality Standards

- **PEP 8 Compliance**: Follows Python coding standards
- **Type Hints**: Comprehensive type annotations on all functions, methods, and variables
- **Docstrings**: Detailed Google-style docstrings for all classes, functions, and methods
- **Inline Comments**: Explicit comments explaining complex logic and execution flow
- **DRY Principle**: No code duplication, reusable components throughout
- **KISS Principle**: Simple, straightforward implementations
- **Clean Architecture**: Strict layer separation with dependency inversion

### Adding New Tools

1. Define input schema in `tools.py` using Pydantic `BaseModel`
2. Implement repository method in `repository.py` with proper validation
3. Create tool function in `tools.py` with `@tool` decorator and error handling
4. Add tool to `AVAILABLE_TOOLS` list in `tools.py`
5. Agent will automatically bind and use the new tool

### Running Tests

Run the complete test suite from the project root:

```bash
uv run pytest tests/ -v -s
```

The repository and tool tests use mocks and an in-memory SQLite database. The agent evaluation tests call the live Unsloth judge configured by `UNSLOTH_BASE_URL` in `.env`, so start that endpoint before running the complete suite. The evaluation tests print each DeepEval metric's score, threshold, status, judge model, and reason; use `-s` to display these reports.

```bash
uv run pytest tests/test_repository.py -v
uv run pytest tests/test_tools.py -v
uv run pytest tests/test_agent_evaluation.py -s
```

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL status
pg_isready

# Verify connection
psql -h localhost -U healthcare -d healthcare -c "SELECT COUNT(*) FROM healthcare_records;"
```

**Common fixes:**
- Ensure PostgreSQL is running
- Verify database and table exist
- Check credentials in `DATABASE_URL`
- Confirm user has read permissions

### LLM Endpoint Issues

- **Endpoint not responding**: Verify Unsloth server is running at `UNSLOTH_BASE_URL`
- **Authentication errors**: Check `UNSLOTH_API_KEY` validity
- **Model not found**: Confirm model name matches available models on endpoint
- **Timeout errors**: Increase timeout or check network connectivity

### Import/Dependency Issues

```bash
# Reinstall dependencies
uv sync --reinstall

# Check Python version
python --version

# Verify package installation
uv run python -c "import healthcare_agent; print('OK')"
```

### Agent Not Using Tools

- Check that tools are properly registered in `AVAILABLE_TOOLS`
- Verify LLM supports tool-calling (most modern models do)
- Enable verbose mode (`AGENT_VERBOSE=true`) to trace tool execution
- Ensure query is clear and requires data retrieval

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 and project style guidelines
- All functions have type hints and docstrings
- Changes are tested before submission
- Commit messages are descriptive

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**Matías J. Magni**  
📧 [matias.magni@gmail.com](mailto:matias.magni@gmail.com)

---

**Built with Clean Architecture, LangChain, LangGraph, and modern Python tooling.**
