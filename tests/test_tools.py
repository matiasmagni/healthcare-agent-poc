"""
Unit Tests for Healthcare Agent Tools.

This module contains comprehensive tests for all LangChain tools that wrap
repository functions. Tests validate tool execution, JSON serialization,
error handling, and proper integration between tools and repository layer.

All tools return JSON-formatted strings for agent consumption, and tests
verify correct formatting and error responses.
"""

import json
from datetime import date
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from tools import (
    AVAILABLE_TOOLS,
    get_patient_billing_summary,
    query_patients_by_condition,
    search_patient_by_name,
)


class TestSearchPatientByNameTool:
    """
    Test suite for search_patient_by_name tool function.
    
    Validates tool correctly calls repository, formats results as JSON,
    and handles errors gracefully.
    """
    
    @pytest.mark.unit
    def test_search_patient_by_name_tool_success(
        self,
        sample_patient_data: List[Dict[str, Any]]
    ) -> None:
        """
        Test successful patient search via tool with results.
        
        Verifies tool calls repository correctly and returns properly
        formatted JSON with success status.
        
        Args:
            sample_patient_data (List[Dict]): Sample patient test data.
        """
        # Arrange: Mock repository response
        test_patient = sample_patient_data[0]
        
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.search_patient_by_name.return_value = [test_patient]
            
            # Act: Call tool
            result_json = search_patient_by_name.invoke({
                "first_name": "John",
                "last_name": "Smith"
            })
            
            # Parse JSON response
            result = json.loads(result_json)
        
        # Assert: Verify response structure
        assert result["status"] == "success", "Status should be success"
        assert result["count"] == 1, "Count should be 1"
        assert len(result["results"]) == 1, "Results should contain 1 patient"
        assert result["results"][0]["id"] == 1001, "Patient ID should match"
        assert "Found" in result["message"], "Message should indicate success"
        
        # Verify repository was called with correct parameters
        mock_repo.search_patient_by_name.assert_called_once_with(
            first_name="John",
            last_name="Smith"
        )
    
    @pytest.mark.unit
    def test_search_patient_by_name_tool_no_results(self) -> None:
        """
        Test tool response when no patients found.
        
        Verifies proper JSON formatting for no results scenario.
        """
        # Arrange
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.search_patient_by_name.return_value = []
            
            # Act
            result_json = search_patient_by_name.invoke({
                "first_name": "Unknown",
                "last_name": "Person"
            })
            result = json.loads(result_json)
        
        # Assert
        assert result["status"] == "no_results", "Status should be no_results"
        assert result["count"] == 0 or "results" in result, "Should indicate no results"
        assert result["results"] == [], "Results should be empty list"
        assert "No patients found" in result["message"]
    
    @pytest.mark.unit
    def test_search_patient_by_name_tool_validation_error(self) -> None:
        """
        Test tool handling of validation errors from repository.
        """
        # Arrange
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.search_patient_by_name.side_effect = ValueError(
                "first_name cannot be empty or whitespace-only"
            )
            
            # Act
            result_json = search_patient_by_name.invoke({
                "first_name": "",
                "last_name": "Smith"
            })
            result = json.loads(result_json)
        
        # Assert
        assert result["status"] == "error", "Status should be error"
        assert result["error_type"] == "validation_error", "Error type should be validation_error"
        assert "cannot be empty" in result["message"].lower()
    
    @pytest.mark.unit
    def test_search_patient_by_name_tool_database_error(self) -> None:
        """
        Test tool handling of database errors.
        """
        # Arrange
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.search_patient_by_name.side_effect = SQLAlchemyError(
                "Connection timeout"
            )
            
            # Act
            result_json = search_patient_by_name.invoke({
                "first_name": "John",
                "last_name": "Smith"
            })
            result = json.loads(result_json)
        
        # Assert
        assert result["status"] == "error", "Status should be error"
        assert result["error_type"] == "database_error", "Error type should be database_error"
        assert "Database error" in result["message"]
    
    @pytest.mark.unit
    def test_search_patient_by_name_tool_json_serialization(self) -> None:
        """
        Test that tool properly serializes date objects in results.
        """
        # Arrange
        test_patient = {
            "id": 1001,
            "name": "John",
            "surname": "Smith",
            "age": 45,
            "gender": "Male",
            "blood_type": "O+",
            "medical_condition": "Diabetes",
            "date_of_admission": date(2024, 1, 15),
            "doctor": "Dr. Johnson",
            "hospital": "City General",
            "insurance_provider": "BlueCross",
            "billing_amount": 12500.50,
            "room_number": 101,
            "admission_type": "Emergency",
            "discharge_date": date(2024, 1, 25),
            "medication": "Metformin",
            "test_results": "Normal"
        }
        
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.search_patient_by_name.return_value = [test_patient]
            
            # Act
            result_json = search_patient_by_name.invoke({
                "first_name": "John",
                "last_name": "Smith"
            })
            
            # Assert - should not raise exception
            result = json.loads(result_json)
            assert result["status"] == "success"


class TestGetPatientBillingSummaryTool:
    """
    Test suite for get_patient_billing_summary tool function.
    """
    
    @pytest.mark.unit
    def test_get_patient_billing_summary_tool_success(self) -> None:
        """
        Test successful billing summary retrieval via tool.
        """
        # Arrange
        billing_summary = {
            "patient_id": 1001,
            "name": "John",
            "surname": "Smith",
            "hospital": "City General Hospital",
            "insurance_provider": "BlueCross",
            "billing_amount": 12500.50,
            "date_of_adision": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 25),
            "length_of_stay_days": 10
        }
        
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.get_patient_billing_summary.return_value = billing_summary
            
            # Act
            result_json = get_patient_billing_summary.invoke({"patient_id": 1001})
            result = json.loads(result_json)
        
        # Assert
        assert result["status"] == "success", "Status should be success"
        assert result["result"]["patient_id"] == 1001
        assert result["result"]["length_of_stay_days"] == 10
        mock_repo.get_patient_billing_summary.assert_called_once_with(patient_id=1001)
    
    @pytest.mark.unit
    def test_get_patient_billing_summary_tool_not_found(self) -> None:
        """
        Test tool response when patient not found.
        """
        # Arrange
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.get_patient_billing_summary.return_value = None
            
            # Act
            result_json = get_patient_billing_summary.invoke({"patient_id": 9999})
            result = json.loads(result_json)
        
        # Assert
        assert result["status"] == "no_results"
        assert "No patient found" in result["message"]
        assert result["result"] is None
    
    @pytest.mark.unit
    def test_get_patient_billing_summary_tool_validation_error(self) -> None:
        """
        Test validation error handling in billing summary tool.
        """
        # Arrange
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.get_patient_billing_summary.side_effect = ValueError(
                "patient_id must be a positive integer"
            )
            
            # Act
            result_json = get_patient_billing_summary.invoke({"patient_id": -1})
            result = json.loads(result_json)
        
        # Assert
        assert result["status"] == "error"
        assert result["error_type"] == "validation_error"


class TestQueryPatientsByConditionTool:
    """
    Test suite for query_patients_by_condition tool function.
    """
    
    @pytest.mark.unit
    def test_query_patients_by_condition_tool_success(self) -> None:
        """
        Test successful condition query via tool.
        """
        # Arrange
        summaries = [
            {
                "id": 1001,
                "name": "John",
                "surname": "Smith",
                "age": 45,
                "gender": "Male",
                "medical_condition": "Diabetes",
                "hospital": "City General",
                "admission_type": "Emergency",
                "doctor": "Dr. Johnson",
                "date_of_admission": date(2024, 1, 15)
            }
        ]
        
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.query_patients_by_condition.return_value = summaries
            
            # Act
            result_json = query_patients_by_condition.invoke({
                "medical_condition": "Diabetes",
                "limit": 5
            })
            result = json.loads(result_json)
        
        # Assert
        assert result["status"] == "success"
        assert result["count"] == 1
        assert result["results"][0]["medical_condition"] == "Diabetes"
        mock_repo.query_patients_by_condition.assert_called_once_with(
            medical_condition="Diabetes",
            limit=5
        )
    
    @pytest.mark.unit
    def test_query_patients_by_condition_tool_default_limit(self) -> None:
        """
        Test tool uses default limit when not specified.
        """
        # Arrange
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.query_patients_by_condition.return_value = []
            
            # Act
            result_json = query_patients_by_condition.invoke({
                "medical_condition": "Hypertension"
            })
            result = json.loads(result_json)
        
        # Assert
        mock_repo.query_patients_by_condition.assert_called_once_with(
            medical_condition="Hypertension",
            limit=5
        )
    
    @pytest.mark.unit
    def test_query_patients_by_condition_tool_no_results(self) -> None:
        """
        Test tool response for condition with no patients.
        """
        # Arrange
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.query_patients_by_condition.return_value = []
            
            # Act
            result_json = query_patients_by_condition.invoke({
                "medical_condition": "Asthma",
                "limit": 10
            })
            result = json.loads(result_json)
        
        # Assert
        assert result["status"] == "no_results"
        assert result["count"] == 0
        assert result["results"] == []
    
    @pytest.mark.unit
    def test_query_patients_by_condition_tool_database_error(self) -> None:
        """
        Test database error handling in condition query tool.
        """
        # Arrange
        with patch('tools.healthcare_repository') as mock_repo:
            mock_repo.query_patients_by_condition.side_effect = SQLAlchemyError(
                "Query failed"
            )
            
            # Act
            result_json = query_patients_by_condition.invoke({
                "medical_condition": "Cancer",
                "limit": 5
            })
            result = json.loads(result_json)
        
        # Assert
        assert result["status"] == "error"
        assert result["error_type"] == "database_error"


class TestAvailableTools:
    """
    Test suite for AVAILABLE_TOOLS list and tool metadata.
    """
    
    @pytest.mark.unit
    def test_available_tools_count(self) -> None:
        """
        Test that exactly 3 tools are available as specified.
        """
        # Arrange & Act
        tools = AVAILABLE_TOOLS
        
        # Assert
        assert len(tools) == 3, "Should have exactly 3 tools"
    
    @pytest.mark.unit
    def test_available_tools_names(self) -> None:
        """
        Test tool names match specifications.
        """
        # Arrange
        expected_tools = [
            "search_patient_by_name",
            "get_patient_billing_summary",
            "query_patients_by_condition"
        ]
        
        # Act
        tool_names = [tool.name for tool in AVAILABLE_TOOLS]
        
        # Assert
        assert tool_names == expected_tools, "Tool names should match specifications"
    
    @pytest.mark.unit
    def test_tools_have_descriptions(self) -> None:
        """
        Test that all tools have proper descriptions.
        """
        # Act & Assert
        for tool in AVAILABLE_TOOLS:
            assert hasattr(tool, 'description'), f"{tool.name} should have description"
            assert len(tool.description) > 0, f"{tool.name} description should not be empty"
            assert hasattr(tool, 'args_schema'), f"{tool.name} should have args_schema"
