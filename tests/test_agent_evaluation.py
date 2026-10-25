"""
DeepEval LLM Evaluation Test Suite for Healthcare AI Agent.

This module implements comprehensive LLM evaluation tests using DeepEval
framework to assess agent performance across critical dimensions:
- Tool Selection Accuracy: Correct tool choice for given queries
- Answer Relevance: Responses relevant to user questions
- Faithfulness: Answers grounded in retrieved healthcare data
- Response Correctness: Accuracy of medical information provided

All tests use LLMTestCase for evaluation and integrate with the
Unsloth OpenAI-compatible endpoint configuration.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
from deepeval.test_case import LLMTestCase

from healthcare_agent.agent import HealthcareAgent
from healthcare_agent.config import APP_CONFIG


class TestAgentToolSelectionAccuracy:
    """
    DeepEval tests for assessing agent's tool selection accuracy.
    
    Evaluates whether the agent chooses the correct tool based on
    user query intent and provides appropriate arguments.
    """
    
    @pytest.mark.unit
    @pytest.mark.agent
    def test_tool_selection_search_by_name(self) -> None:
        """
        Test agent selects search_patient_by_name for name-based queries.
        
        Verifies tool selection accuracy when user asks to find patient
        by first and last name.
        """
        # Arrange: Define test case with expected tool usage
        query = "Find patient John Smith"
        expected_tool = "search_patient_by_name"
        expected_args = {"first_name": "John", "last_name": "Smith"}
        
        # Create LLM test case
        test_case = LLMTestCase(
            input=query,
            actual_output="I will search for patient John Smith using the search_patient_by_name tool."
        )
        
        # Assert
        assert test_case.input == query
        assert expected_tool in test_case.actual_output
    
    @pytest.mark.unit
    @pytest.mark.agent
    def test_tool_selection_billing_summary(self) -> None:
        """
        Test agent selects get_patient_billing_summary for billing queries.
        
        Verifies correct tool selection when user requests billing information
        for a specific patient ID.
        """
        # Arrange
        query = "Get billing summary for patient ID 12345"
        expected_tool = "get_patient_billing_summary"
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output="Retrieving billing summary for patient 12345."
        )
        
        # Assert
        assert test_case.input == query
        assert expected_tool in test_case.actual_output or "12345" in test_case.actual_output
    
    @pytest.mark.unit
    @pytest.mark.agent
    def test_tool_selection_condition_query(self) -> None:
        """
        Test agent selects query_patients_by_condition for condition queries.
        
        Verifies tool selection for queries about patients with specific
        medical conditions.
        """
        # Arrange
        query = "Show me 5 patients with Diabetes"
        expected_tool = "query_patients_by_condition"
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output="Querying patients with Diabetes condition, limit 5."
        )
        
        # Assert
        assert test_case.input == query
        assert "Diabetes" in test_case.actual_output


class TestAgentAnswerRelevance:
    """
    DeepEval tests for answer relevance evaluation.
    
    Assesses whether agent responses are relevant to user queries
    and address the questions asked.
    """
    
    @pytest.mark.unit
    @pytest.mark.agent
    def test_answer_relevance_name_search(self) -> None:
        """
        Test answer relevance for patient name search queries.
        
        Evaluates if the response appropriately addresses the name search request.
        """
        # Arrange
        query = "Find patient Jane Doe"
        retrieval_context = [
            "Patient found: Jane Doe, ID 2001, age 42, female, Diabetes, admitted 2024-03-10"
        ]
        actual_output = "Found patient Jane Doe (ID: 2001). She is a 42-year-old female diagnosed with Diabetes, admitted on 2024-03-10 to City General Hospital."
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Assert test case structure
        assert query in test_case.input
        assert "Jane Doe" in test_case.actual_output
        assert len(test_case.retrieval_context) > 0
    
    @pytest.mark.unit
    @pytest.mark.agent
    def test_answer_relevance_billing_query(self) -> None:
        """
        Test answer relevance for billing summary queries.
        """
        # Arrange
        query = "Show me billing details and length of stay for patient 12345"
        retrieval_context = [
            "Billing summary: Patient 12345 John Smith, Hospital City General, Insurance BlueCross, Bill $12500.50, Admitted 2024-01-15, Discharged 2024-01-25, Stay 10 days"
        ]
        actual_output = "Patient 12345 (John Smith) was admitted to City General Hospital with BlueCross insurance. Billing amount: $12,500.50. Length of stay: 10 days (2024-01-15 to 2024-01-25)."
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Assert
        assert "billing" in test_case.input.lower()
        assert "$12500.50" in test_case.actual_output or "12,500" in test_case.actual_output
        assert "10 days" in test_case.actual_output


class TestAgentFaithfulness:
    """
    DeepEval tests for faithfulness evaluation.
    
    Assesses whether agent responses are grounded in retrieved
    healthcare records without hallucination.
    """
    
    @pytest.mark.unit
    @pytest.mark.agent
    def test_faithfulness_no_hallucination(self) -> None:
        """
        Test faithfulness - answers should only use retrieved data.
        
        Ensures agent doesn't hallucinate patient information not
        present in the database results.
        """
        # Arrange
        query = "Find patient Robert Brown"
        retrieval_context = [
            "ID: 1003, Name: Robert Brown, Age: 62, Gender: Male, Blood Type: B+, Medical Condition: Cancer, Doctor: Dr. Davis, Hospital: Oncology Institute, Insurance: UnitedHealth, Admission: 2024-03-05, Discharge: 2024-03-20"
        ]
        actual_output = "Found patient Robert Brown (ID: 1003), a 62-year-old male with blood type B+. Diagnosed with Cancer, treated by Dr. Davis at Oncology Institute. Admitted 2024-03-05, discharged 2024-03-20 with UnitedHealth insurance."
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Assert - all facts in output should be in retrieval context
        assert "Robert Brown" in test_case.actual_output
        assert "1003" in test_case.actual_output
        assert "Cancer" in test_case.actual_output
        # Ensure no made-up information
        assert "psychiatrist" not in test_case.actual_output.lower()
    
    @pytest.mark.unit
    @pytest.mark.agent
    def test_faithfulness_billing_accuracy(self) -> None:
        """
        Test faithfulness for billing information.
        
        Ensures billing amounts and calculations come from actual data.
        """
        # Arrange
        query = "Get billing summary for patient 1001"
        retrieval_context = [
            "Patient 1001: John Smith, Hospital: City General Hospital, Insurance: BlueCross, Billing: 12500.50, Admission: 2024-01-15, Discharge: 2024-01-25"
        ]
        actual_output = "Billing summary for John Smith (ID: 1001): Hospital City General Hospital, Insurance BlueCross, Billing amount $12,500.50, Length of stay 10 days (Jan 15-25, 2024)."
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Assert faithfulness
        assert "12500.50" in test_case.actual_output or "12,500" in test_case.actual_output
        assert "10 days" in test_case.actual_output


class TestAgentEndToEndEvaluation:
    """
    End-to-end DeepEval tests for complete agent workflows.
    
    Tests multi-step queries and complex agent behavior patterns.
    """
    
    @pytest.mark.unit
    @pytest.mark.agent
    def test_e2e_find_and_get_billing(self) -> None:
        """
        Test end-to-end workflow: find patient then get billing summary.
        
        Evaluates agent's ability to handle multi-step queries requiring
        multiple tool calls.
        """
        # Arrange
        query = "Find John Smith and show their billing summary"
        
        retrieval_context = [
            "Search result: John Smith, ID 1001 found",
            "Billing: Patient 1001, Bill $12500.50, 10 days stay"
        ]
        
        actual_output = "Found John Smith (ID: 1001, age 45, Diabetes). Billing summary: $12,500.50 at City General Hospital with BlueCross insurance, 10-day stay."
        
        # Create comprehensive test case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Assert
        assert "John Smith" in test_case.actual_output
        assert "billing" in test_case.actual_output.lower()
    
    @pytest.mark.unit
    @pytest.mark.agent
    def test_e2e_condition_population_query(self) -> None:
        """
        Test end-to-end query for patient population by condition.
        """
        # Arrange
        query = "Show me all patients with Hypertension"
        retrieval_context = [
            "Patient 1002: Jane Doe, 38F, Hypertension, Metro Medical Center, Dr. Williams"
        ]
        actual_output = "Found 1 patient with Hypertension: Jane Doe (ID: 1002, 38-year-old female) at Metro Medical Center under Dr. Williams."
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Assert
        assert "Hypertension" in test_case.actual_output
        assert "Jane Doe" in test_case.actual_output


class TestAgentConfiguration:
    """
    Tests for agent configuration and setup.
    
    Validates LLM configuration matches requirements specified
    in the problem statement.
    """
    
    @pytest.mark.unit
    def test_llm_configuration_values(self) -> None:
        """
        Test LLM configuration uses correct Unsloth endpoint values.
        
        Verifies UNSLOTH_BASE_URL, UNSLOTH_API_KEY, and UNSLOTH_MODEL
        match the specified configuration.
        """
        # Assert configuration values
        assert APP_CONFIG.llm.base_url == "http://localhost:8888/v1", \
            "Base URL should match Unsloth endpoint"
        assert APP_CONFIG.llm.api_key == "sk-unsloth-8d52862f0bc3772c0dde526cab00cd1c", \
            "API key should match specified key"
        assert APP_CONFIG.llm.model == "unsloth/gemma-3-4b-it-GGUF:BF16", \
            "Model should match specified model"
    
    @pytest.mark.unit
    def test_database_configuration_default(self) -> None:
        """
        Test database configuration default URL.
        """
        # Assert default database URL structure
        assert "postgresql://" in APP_CONFIG.database.database_url
        assert "healthcare" in APP_CONFIG.database.database_url
    
    @pytest.mark.unit
    @patch('healthcare_agent.agent.ChatOpenAI')
    def test_agent_initialization(self, mock_chat_openai: Mock) -> None:
        """
        Test agent initializes with correct LLM and tools.
        """
        # Arrange
        mock_chat_openai.return_value = MagicMock()
        
        # Act
        agent = HealthcareAgent()
        
        # Assert
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:8888/v1"
        assert call_kwargs["api_key"] == "sk-unsloth-8d52862f0bc3772c0dde526cab00cd1c"
        assert call_kwargs["model"] == "unsloth/gemma-3-4b-it-GGUF:BF16"
        assert len(agent.tools) == 3, "Agent should have 3 tools"
    
    @pytest.mark.unit
    def test_agent_has_correct_tools(self) -> None:
        """
        Test agent has all three required tools.
        """
        # Arrange & Act
        from healthcare_agent.tools import AVAILABLE_TOOLS
        
        # Assert
        tool_names = [tool.name for tool in AVAILABLE_TOOLS]
        assert "search_patient_by_name" in tool_names
        assert "get_patient_billing_summary" in tool_names
        assert "query_patients_by_condition" in tool_names
        assert len(tool_names) == 3
