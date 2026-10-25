"""
Unit Tests for HealthcareRepository Data Access Layer.

This module contains comprehensive unit tests for all repository methods
interacting with the healthcare_records table. Tests cover normal operations,
edge cases, validation errors, and database error handling using mocked
database sessions for complete isolation.

All tests follow AAA pattern (Arrange, Act, Assert) with detailed docstrings
and inline comments explaining test logic and expected behavior.
"""

import json
from datetime import date
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from healthcare_agent.repository import HealthcareRepository


class TestSearchPatientByName:
    """
    Test suite for search_patient_by_name method.
    
    Tests cover case-insensitive search, validation, multiple results,
    and error handling scenarios.
    """
    
    @pytest.mark.unit
    def test_search_patient_by_name_success(
        self,
        mock_db_manager: Mock,
        sample_patient_data: List[Dict[str, Any]]
    ) -> None:
        """
        Test successful patient search by name with matching results.
        
        Verifies that search_patient_by_name correctly queries the database,
        normalizes input, and returns patient records as dictionaries.
        
        Args:
            mock_db_manager (Mock): Mocked database manager fixture.
            sample_patient_data (List[Dict]): Sample patient test data.
        """
        # Arrange: Setup test data and mock database responses
        repo = HealthcareRepository()
        test_patient = sample_patient_data[0]  # John Smith
        
        # Create mock row object with all required attributes
        mock_row = MagicMock()
        mock_row.id = test_patient["id"]
        mock_row.name = test_patient["name"]
        mock_row.surname = test_patient["surname"]
        mock_row.age = test_patient["age"]
        mock_row.gender = test_patient["gender"]
        mock_row.blood_type = test_patient["blood_type"]
        mock_row.medical_condition = test_patient["medical_condition"]
        mock_row.date_of_admission = test_patient["date_of_admission"]
        mock_row.doctor = test_patient["doctor"]
        mock_row.hospital = test_patient["hospital"]
        mock_row.insurance_provider = test_patient["insurance_provider"]
        mock_row.billing_amount = test_patient["billing_amount"]
        mock_row.room_number = test_patient["room_number"]
        mock_row.admission_type = test_patient["admission_type"]
        mock_row.discharge_date = test_patient["discharge_date"]
        mock_row.medication = test_patient["medication"]
        mock_row.test_results = test_patient["test_results"]
        
        # Create mock session and result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        
        # Patch db_manager to use mock session
        with patch('healthcare_agent.repository.db_manager') as mock_db:
            # Setup context manager
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = None
            
            # Act: Execute search
            result = repo.search_patient_by_name("John", "Smith")
        
        # Assert: Verify results
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1, "Should return exactly one patient"
        assert result[0]["id"] == 1001, "Patient ID should match"
        assert result[0]["name"] == "John", "First name should match"
        assert result[0]["surname"] == "Smith", "Last name should match"
        assert result[0]["medical_condition"] == "Diabetes", "Medical condition should match"
        assert isinstance(result[0]["billing_amount"], float), "Billing amount should be converted to float"
        
        # Verify database query was executed with correct parameters
        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args
        assert call_args[0][1]["first_name"] == "John"
        assert call_args[0][1]["last_name"] == "Smith"
    
    @pytest.mark.unit
    def test_search_patient_by_name_no_results(
        self,
        mock_db_manager: Mock
    ) -> None:
        """
        Test patient search returning empty list when no matches found.
        
        Verifies that the method correctly handles no results scenario
        and returns an empty list as specified.
        
        Args:
            mock_db_manager (Mock): Mocked database manager fixture.
        """
        # Arrange: Setup empty result
        repo = HealthcareRepository()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []  # No matches
        mock_session.execute.return_value = mock_result
        
        # Patch db_manager
        with patch('healthcare_agent.repository.db_manager') as mock_db:
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = None
            
            # Act: Search for non-existent patient
            result = repo.search_patient_by_name("NonExistent", "Patient")
        
        # Assert: Verify empty result
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 0, "Should return empty list when no matches found"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("first_name,last_name", [
        ("", "Smith"),
        ("   ", "Doe"),
        ("John", ""),
        ("Jane", "   "),
        ("", ""),
    ])
    def test_search_patient_by_name_validation_errors(
        self,
        first_name: str,
        last_name: str
    ) -> None:
        """
        Test input validation for empty or whitespace-only names.
        
        Verifies that ValueError is raised for invalid inputs as per
        repository method specifications.
        
        Args:
            first_name (str): Test first name (invalid cases).
            last_name (str): Test last name (invalid cases).
        """
        # Arrange: Create repository instance
        repo = HealthcareRepository()
        
        # Act & Assert: Verify validation error is raised
        with pytest.raises(ValueError) as exc_info:
            repo.search_patient_by_name(first_name, last_name)
        
        # Verify error message contains expected text
        error_message = str(exc_info.value)
        assert "cannot be empty" in error_message.lower()
    
    @pytest.mark.unit
    def test_search_patient_by_name_database_error(
        self,
        mock_db_manager: Mock
    ) -> None:
        """
        Test database error handling during patient search.
        
        Verifies that SQLAlchemy errors are properly caught and re-raised
        with descriptive error messages.
        
        Args:
            mock_db_manager (Mock): Mocked database manager fixture.
        """
        # Arrange: Setup database error
        repo = HealthcareRepository()
        mock_session = MagicMock()
        # Simulate database error
        mock_session.execute.side_effect = SQLAlchemyError("Connection failed")
        
        # Patch db_manager
        with patch('healthcare_agent.repository.db_manager') as mock_db:
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = None
            
            # Act & Assert: Verify database error is re-raised
            with pytest.raises(SQLAlchemyError) as exc_info:
                repo.search_patient_by_name("John", "Smith")
            
            # Verify error message includes context
            assert "Failed to search patients by name" in str(exc_info.value)


class TestGetPatientBillingSummary:
    """
    Test suite for get_patient_billing_summary method.
    
    Tests cover successful retrieval, not found scenarios, length of stay
    calculation, and validation errors.
    """
    
    @pytest.mark.unit
    def test_get_patient_billing_summary_success(
        self,
        sample_patient_data: List[Dict[str, Any]]
    ) -> None:
        """
        Test successful billing summary retrieval with length of stay calculation.
        
        Verifies that billing summary is correctly retrieved and length of stay
        is calculated from admission and discharge dates.
        
        Args:
            sample_patient_data (List[Dict]): Sample patient test data.
        """
        # Arrange: Setup test patient with known dates
        repo = HealthcareRepository()
        test_patient = sample_patient_data[0]
        # John Smith: admitted 2024-01-15, discharged 2024-01-25 -> 10 days stay
        
        mock_row = MagicMock()
        mock_row.id = test_patient["id"]
        mock_row.name = test_patient["name"]
        mock_row.surname = test_patient["surname"]
        mock_row.hospital = test_patient["hospital"]
        mock_row.insurance_provider = test_patient["insurance_provider"]
        mock_row.billing_amount = test_patient["billing_amount"]
        mock_row.date_of_admission = test_patient["date_of_admission"]
        mock_row.discharge_date = test_patient["discharge_date"]
        
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result
        
        # Patch db_manager
        with patch('healthcare_agent.repository.db_manager') as mock_db:
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = None
            
            # Act: Get billing summary
            result = repo.get_patient_billing_summary(1001)
        
        # Assert: Verify billing summary
        assert result is not None, "Result should not be None for existing patient"
        assert result["patient_id"] == 1001, "Patient ID should match"
        assert result["name"] == "John", "Name should match"
        assert result["surname"] == "Smith", "Surname should match"
        assert result["hospital"] == "City General Hospital", "Hospital should match"
        assert result["insurance_provider"] == "BlueCross", "Insurance provider should match"
        assert result["billing_amount"] == 12500.50, "Billing amount should match"
        assert result["length_of_stay_days"] == 10, "Length of stay should be 10 days"
        assert isinstance(result["billing_amount"], float), "Billing amount should be float"
    
    @pytest.mark.unit
    def test_get_patient_billing_summary_not_found(self) -> None:
        """
        Test billing summary retrieval for non-existent patient.
        
        Verifies that None is returned when patient is not found.
        """
        # Arrange
        repo = HealthcareRepository()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None  # Patient not found
        mock_session.execute.return_value = mock_result
        
        # Patch db_manager
        with patch('healthcare_agent.repository.db_manager') as mock_db:
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = None
            
            # Act
            result = repo.get_patient_billing_summary(9999)
        
        # Assert
        assert result is None, "Should return None for non-existent patient"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_id", [0, -1, -100])
    def test_get_patient_billing_summary_validation_errors(
        self,
        invalid_id: int
    ) -> None:
        """
        Test validation for invalid patient IDs.
        
        Verifies that ValueError is raised for non-positive integers.
        
        Args:
            invalid_id (int): Invalid patient ID values.
        """
        # Arrange
        repo = HealthcareRepository()
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            repo.get_patient_billing_summary(invalid_id)
        
        assert "positive integer" in str(exc_info.value).lower()
    
    @pytest.mark.unit
    def test_get_patient_billing_summary_database_error(self) -> None:
        """
        Test database error handling for billing summary retrieval.
        """
        # Arrange
        repo = HealthcareRepository()
        mock_session = MagicMock()
        mock_session.execute.side_effect = SQLAlchemyError("Database error")
        
        # Patch db_manager
        with patch('healthcare_agent.repository.db_manager') as mock_db:
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = None
            
            # Act & Assert
            with pytest.raises(SQLAlchemyError) as exc_info:
                repo.get_patient_billing_summary(1001)
            
            assert "Failed to retrieve billing summary" in str(exc_info.value)


class TestQueryPatientsByCondition:
    """
    Test suite for query_patients_by_condition method.
    
    Tests cover condition filtering, result limits, validation,
    and error handling.
    """
    
    @pytest.mark.unit
    def test_query_patients_by_condition_success(
        self,
        sample_patient_data: List[Dict[str, Any]]
    ) -> None:
        """
        Test successful query for patients by medical condition.
        
        Verifies filtering by condition and correct summary data structure.
        
        Args:
            sample_patient_data (List[Dict]): Sample patient test data.
        """
        # Arrange
        repo = HealthcareRepository()
        # Find all patients with Diabetes
        diabetes_patients = [p for p in sample_patient_data if p["medical_condition"] == "Diabetes"]
        
        mock_rows = []
        for patient in diabetes_patients:
            mock_row = MagicMock()
            mock_row.id = patient["id"]
            mock_row.name = patient["name"]
            mock_row.surname = patient["surname"]
            mock_row.age = patient["age"]
            mock_row.gender = patient["gender"]
            mock_row.medical_condition = patient["medical_condition"]
            mock_row.hospital = patient["hospital"]
            mock_row.admission_type = patient["admission_type"]
            mock_row.doctor = patient["doctor"]
            mock_row.date_of_admission = patient["date_of_admission"]
            mock_rows.append(mock_row)
        
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result
        
        # Patch db_manager
        with patch('healthcare_agent.repository.db_manager') as mock_db:
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = None
            
            # Act
            result = repo.query_patients_by_condition("Diabetes", limit=5)
        
        # Assert
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == len(diabetes_patients), "Should return matching patients"
        assert result[0]["medical_condition"] == "Diabetes", "Condition should match"
        assert "age" in result[0], "Summary should include age"
        assert "hospital" in result[0], "Summary should include hospital"
        assert "doctor" in result[0], "Summary should include doctor"
        
        # Verify limit was passed
        call_args = mock_session.execute.call_args
        assert call_args[0][1]["limit"] == 5
    
    @pytest.mark.unit
    def test_query_patients_by_condition_no_results(self) -> None:
        """
        Test query returning empty list for condition with no patients.
        """
        # Arrange
        repo = HealthcareRepository()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        # Patch db_manager
        with patch('healthcare_agent.repository.db_manager') as mock_db:
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = None
            
            # Act
            result = repo.query_patients_by_condition("RareCondition", limit=5)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 0
    
    @pytest.mark.unit
    @pytest.mark.parametrize("limit", [0, -1, 101, 200])
    def test_query_patients_by_condition_invalid_limit(self, limit: int) -> None:
        """
        Test validation for invalid limit values.
        
        Limit must be between 1 and 100 inclusive.
        
        Args:
            limit (int): Invalid limit values.
        """
        # Arrange
        repo = HealthcareRepository()
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            repo.query_patients_by_condition("Diabetes", limit=limit)
        
        assert "limit must be" in str(exc_info.value).lower()
    
    @pytest.mark.unit
    def test_query_patients_by_condition_empty_condition(self) -> None:
        """
        Test validation for empty medical condition.
        """
        # Arrange
        repo = HealthcareRepository()
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            repo.query_patients_by_condition("", limit=5)
        
        assert "medical_condition cannot be empty" in str(exc_info.value).lower()
    
    @pytest.mark.unit
    def test_query_patients_by_condition_default_limit(self) -> None:
        """
        Test that default limit is 5 when not specified.
        """
        # Arrange
        repo = HealthcareRepository()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        # Patch db_manager
        with patch('healthcare_agent.repository.db_manager') as mock_db:
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = None
            
            # Act
            result = repo.query_patients_by_condition("Diabetes")
        
        # Assert - verify default limit of 5 was used
        call_args = mock_session.execute.call_args
        assert call_args[0][1]["limit"] == 5
