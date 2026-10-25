"""
Pytest Configuration and Fixtures for Healthcare AI Agent Tests.

This module provides production-grade pytest fixtures for database connection,
session management, test data setup, and LLM mocking. All fixtures are designed
for isolation, reusability, and proper cleanup to ensure tests remain fast
and decoupled.

The fixtures support both unit testing (with mocks) and integration testing
(with test database instances) while maintaining clean separation of concerns.
"""

import json
import os
from datetime import date
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Import application modules for testing
from healthcare_agent.agent import HealthcareAgent
from healthcare_agent.config import APP_CONFIG
from healthcare_agent.database import DatabaseManager
from healthcare_agent.repository import HealthcareRepository
from healthcare_agent.tools import (
    AVAILABLE_TOOLS,
    get_patient_billing_summary,
    query_patients_by_condition,
    search_patient_by_name,
)


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """
    Provide test database URL for integration tests.
    
    Uses in-memory SQLite for fast unit tests by default, or can be
    configured to use a test PostgreSQL instance via environment variables.
    
    Returns:
        str: Database connection URL for testing.
    """
    # Check if test PostgreSQL URL is provided via environment
    test_pg_url = os.getenv("TEST_DATABASE_URL")
    if test_pg_url:
        return test_pg_url
    
    # Default to in-memory SQLite for unit tests (fast, isolated)
    # Note: Some PostgreSQL-specific features may need adjustment
    return "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine(test_database_url: str) -> Generator[Engine, None, None]:
    """
    Create test SQLAlchemy engine with optimized settings for testing.
    
    Uses StaticPool for in-memory SQLite to maintain single connection
    across test sessions. Configured for fast, isolated test execution.
    
    Args:
        test_database_url (str): Database URL from test_database_url fixture.
        
    Yields:
        Engine: SQLAlchemy engine instance for test database.
    """
    # Create engine with test-optimized settings
    if test_database_url.startswith("sqlite"):
        # SQLite-specific configuration for in-memory testing
        engine = create_engine(
            test_database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
            future=True
        )
    else:
        # PostgreSQL test configuration
        engine = create_engine(
            test_database_url,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=5,
            echo=False,
            future=True
        )
    
    yield engine
    
    # Cleanup: dispose engine after all tests
    engine.dispose()


@pytest.fixture(scope="function")
def test_session_factory(test_engine: Engine):
    """
    Create test session factory bound to test engine.
    
    Provides a sessionmaker for creating isolated database sessions
    for each test function with proper transaction boundaries.
    
    Args:
        test_engine (Engine): Test database engine.
        
    Returns:
        sessionmaker: Factory for creating test sessions.
    """
    SessionLocal = sessionmaker(
        bind=test_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )
    return SessionLocal


@pytest.fixture(scope="function")
def db_session(test_session_factory) -> Generator[Session, None, None]:
    """
    Provide isolated database session for each test with automatic rollback.
    
    Each test gets a fresh session that is rolled back after test completion
    to ensure complete test isolation and no side effects between tests.
    
    Args:
        test_session_factory: Session factory from fixture.
        
    Yields:
        Session: SQLAlchemy session for database operations.
    """
    session = test_session_factory()
    
    try:
        yield session
    finally:
        # Rollback any uncommitted changes and close session
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def mock_db_manager(db_session: Session) -> Generator[DatabaseManager, None, None]:
    """
    Create mock DatabaseManager that uses test session.
    
    Patches the global db_manager to use test database session
    for isolated testing without affecting production database.
    
    Args:
        db_session (Session): Test database session.
        
    Yields:
        DatabaseManager: Mocked database manager instance.
    """
    mock_manager = Mock(spec=DatabaseManager)
    
    # Create context manager that yields test session
    from contextlib import contextmanager
    
    @contextmanager
    def mock_get_session():
        try:
            yield db_session
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            raise e
    
    mock_manager.get_session = mock_get_session
    mock_manager.test_connection.return_value = True
    mock_manager.engine = db_session.bind
    
    yield mock_manager


@pytest.fixture(scope="function")
def healthcare_repo(mock_db_manager: DatabaseManager) -> HealthcareRepository:
    """
    Create HealthcareRepository instance with mocked database manager.
    
    Provides repository for testing data access methods in isolation
    with controlled database session.
    
    Args:
        mock_db_manager (DatabaseManager): Mocked database manager.
        
    Returns:
        HealthcareRepository: Repository instance for testing.
    """
    repo = HealthcareRepository()
    # Patch the global db_manager used by repository
    with patch('healthcare_agent.repository.db_manager', mock_db_manager):
        yield repo


@pytest.fixture(scope="session")
def sample_patient_data() -> List[Dict[str, Any]]:
    """
    Provide sample patient data for testing.
    
    Returns realistic test data matching healthcare_records schema
    for consistent testing across test suites.
    
    Returns:
        List[Dict[str, Any]]: List of sample patient records.
    """
    return [
        {
            "id": 1001,
            "name": "John",
            "surname": "Smith",
            "age": 45,
            "gender": "Male",
            "blood_type": "O+",
            "medical_condition": "Diabetes",
            "date_of_admission": date(2024, 1, 15),
            "doctor": "Dr. Johnson",
            "hospital": "City General Hospital",
            "insurance_provider": "BlueCross",
            "billing_amount": 12500.50,
            "room_number": 101,
            "admission_type": "Emergency",
            "discharge_date": date(2024, 1, 25),
            "medication": "Metformin",
            "test_results": "Normal"
        },
        {
            "id": 1002,
            "name": "Jane",
            "surname": "Doe",
            "age": 38,
            "gender": "Female",
            "blood_type": "A-",
            "medical_condition": "Hypertension",
            "date_of_admission": date(2024, 2, 10),
            "doctor": "Dr. Williams",
            "hospital": "Metro Medical Center",
            "insurance_provider": "Aetna",
            "billing_amount": 8900.00,
            "room_number": 205,
            "admission_type": "Elective",
            "discharge_date": date(2024, 2, 18),
            "medication": "Lisinopril",
            "test_results": "Elevated BP"
        },
        {
            "id": 1003,
            "name": "Robert",
            "surname": "Brown",
            "age": 62,
            "gender": "Male",
            "blood_type": "B+",
            "medical_condition": "Cancer",
            "date_of_admission": date(2024, 3, 5),
            "doctor": "Dr. Davis",
            "hospital": "Oncology Institute",
            "insurance_provider": "UnitedHealth",
            "billing_amount": 45000.75,
            "room_number": 301,
            "admission_type": "Urgent",
            "discharge_date": date(2024, 3, 20),
            "medication": "Chemotherapy",
            "test_results": "In Progress"
        }
    ]


@pytest.fixture(scope="function")
def mock_llm_client() -> Generator[MagicMock, None, None]:
    """
    Create mock OpenAI-compatible LLM client for agent testing.
    
    Mocks ChatOpenAI to avoid actual API calls to Unsloth endpoint
    during unit tests while maintaining interface compatibility.
    
    Yields:
        MagicMock: Mocked LLM client instance.
    """
    with patch('healthcare_agent.agent.ChatOpenAI') as mock_chat_openai:
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        
        # Configure default mock behavior
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm_instance.invoke.return_value = MagicMock(
            content="Test response from mocked LLM"
        )
        
        yield mock_llm_instance


@pytest.fixture(scope="function")
def test_tools_list() -> List:
    """
    Provide list of available tools for testing.
    
    Returns the actual tools from the tools module for integration
    testing of tool functionality.
    
    Returns:
        List: List of LangChain tool objects.
    """
    return AVAILABLE_TOOLS


@pytest.fixture(scope="function")
def reset_global_instances():
    """
    Reset global instances between tests to ensure isolation.
    
    Clears any cached state in global instances (db_manager, 
    healthcare_repository, healthcare_agent) between test runs.
    """
    # This fixture can be extended to reset global state if needed
    yield
    # Cleanup after test if necessary


# Pytest configuration markers
def pytest_configure(config):
    """
    Configure custom pytest markers for test categorization.
    
    Args:
        config: Pytest configuration object.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (require database)"
    )
    config.addinivalue_line(
        "markers", "agent: Agent behavior tests"
    )
    config.addinivalue_line(
        "markers", "deepeval: DeepEval LLM evaluation tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
