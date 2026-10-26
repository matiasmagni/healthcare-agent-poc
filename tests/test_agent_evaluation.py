"""
DeepEval metric tests for healthcare agent behavior.

The official DeepEval metrics score representative tool-selection, answer
relevance, and faithfulness cases. LLM judge outputs are stubbed so these unit
tests remain deterministic and do not require a running inference endpoint.
"""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ToolCorrectnessMetric,
)
from deepeval.metrics.answer_relevancy.schema import AnswerRelevancyVerdict
from deepeval.metrics.base_metric import Verdict
from deepeval.metrics.faithfulness.schema import FaithfulnessVerdict
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

from healthcare_agent.agent import HealthcareAgent
from healthcare_agent.config import APP_CONFIG

METRIC_THRESHOLD = 0.7


class OfflineMetricJudge(DeepEvalBaseLLM):
    """Prevent DeepEval from initializing a network-backed default judge."""

    def load_model(self, *args: Any, **kwargs: Any) -> "OfflineMetricJudge":
        return self

    def generate(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("The test must provide its deterministic judge verdicts")

    async def a_generate(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("The test must provide its deterministic judge verdicts")

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return "offline-test-judge"


def measure_tool_correctness(test_case: LLMTestCase) -> ToolCorrectnessMetric:
    """Run DeepEval tool correctness against expected tool names and arguments."""
    metric = ToolCorrectnessMetric(
        threshold=METRIC_THRESHOLD,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        async_mode=False,
        should_exact_match=True,
        model=OfflineMetricJudge(model="offline-test-judge"),
    )
    metric.measure(test_case, _show_indicator=False)
    return metric


def measure_answer_relevance(
    test_case: LLMTestCase,
    monkeypatch: pytest.MonkeyPatch,
    verdict: Verdict,
) -> AnswerRelevancyMetric:
    """Measure relevance with a deterministic stub for the LLM judge."""
    metric = AnswerRelevancyMetric(
        threshold=METRIC_THRESHOLD,
        async_mode=False,
        include_reason=False,
        model=OfflineMetricJudge(model="offline-test-judge"),
    )
    monkeypatch.setattr(
        metric,
        "_generate_statements",
        lambda actual_output, multimodal: [actual_output],
    )
    monkeypatch.setattr(
        metric,
        "_generate_verdicts",
        lambda input, multimodal: [
            AnswerRelevancyVerdict(verdict=verdict)
        ],
    )
    metric.measure(test_case, _show_indicator=False)
    return metric


def measure_faithfulness(
    test_case: LLMTestCase,
    monkeypatch: pytest.MonkeyPatch,
    verdicts: list[Verdict],
) -> FaithfulnessMetric:
    """Measure faithfulness with deterministic LLM-judge verdicts."""
    metric = FaithfulnessMetric(
        threshold=METRIC_THRESHOLD,
        async_mode=False,
        include_reason=False,
        model=OfflineMetricJudge(model="offline-test-judge"),
    )
    claims = [test_case.actual_output or "" for _ in verdicts]
    monkeypatch.setattr(
        metric,
        "_generate_truths",
        lambda retrieval_context, multimodal: retrieval_context,
    )
    monkeypatch.setattr(
        metric,
        "_generate_claims",
        lambda actual_output, multimodal: claims,
    )
    monkeypatch.setattr(
        metric,
        "_generate_verdicts",
        lambda multimodal: [
            FaithfulnessVerdict(verdict=verdict) for verdict in verdicts
        ],
    )
    metric.measure(test_case, _show_indicator=False)
    return metric


class TestAgentToolSelectionAccuracy:
    """Evaluate tool selection and arguments with DeepEval's tool metric."""

    @pytest.mark.unit
    @pytest.mark.agent
    def test_tool_selection_search_by_name(self) -> None:
        """Score the expected patient-search tool call and its arguments."""
        test_case = LLMTestCase(
            input="Find patient John Smith",
            actual_output="Patient search completed.",
            tools_called=[
                ToolCall(
                    name="search_patient_by_name",
                    input_parameters={
                        "first_name": "John",
                        "last_name": "Smith",
                    },
                )
            ],
            expected_tools=[
                ToolCall(
                    name="search_patient_by_name",
                    input_parameters={
                        "first_name": "John",
                        "last_name": "Smith",
                    },
                )
            ],
        )

        metric = measure_tool_correctness(test_case)

        assert metric.score == 1.0
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_tool_selection_billing_summary(self) -> None:
        """Score the expected billing-summary tool call and patient ID."""
        test_case = LLMTestCase(
            input="Get billing summary for patient ID 12345",
            actual_output="Billing summary retrieved.",
            tools_called=[
                ToolCall(
                    name="get_patient_billing_summary",
                    input_parameters={"patient_id": 12345},
                )
            ],
            expected_tools=[
                ToolCall(
                    name="get_patient_billing_summary",
                    input_parameters={"patient_id": 12345},
                )
            ],
        )

        metric = measure_tool_correctness(test_case)

        assert metric.score == 1.0
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_tool_selection_condition_query(self) -> None:
        """Score condition-query tool selection, condition, and requested limit."""
        test_case = LLMTestCase(
            input="Show me 5 patients with Diabetes",
            actual_output="Condition query completed.",
            tools_called=[
                ToolCall(
                    name="query_patients_by_condition",
                    input_parameters={
                        "medical_condition": "Diabetes",
                        "limit": 5,
                    },
                )
            ],
            expected_tools=[
                ToolCall(
                    name="query_patients_by_condition",
                    input_parameters={
                        "medical_condition": "Diabetes",
                        "limit": 5,
                    },
                )
            ],
        )

        metric = measure_tool_correctness(test_case)

        assert metric.score == 1.0
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_wrong_tool_selection_fails_threshold(self) -> None:
        """Reject an incorrect tool even when a tool was called."""
        test_case = LLMTestCase(
            input="Find patient John Smith",
            actual_output="Billing summary retrieved.",
            tools_called=[
                ToolCall(
                    name="get_patient_billing_summary",
                    input_parameters={"patient_id": 12345},
                )
            ],
            expected_tools=[
                ToolCall(
                    name="search_patient_by_name",
                    input_parameters={
                        "first_name": "John",
                        "last_name": "Smith",
                    },
                )
            ],
        )

        metric = measure_tool_correctness(test_case)

        assert metric.score < METRIC_THRESHOLD
        assert metric.success is False


class TestAgentAnswerRelevance:
    """Evaluate whether agent answers address the user's question."""

    @pytest.mark.unit
    @pytest.mark.agent
    def test_answer_relevance_name_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pass a relevant answer to a patient search request."""
        test_case = LLMTestCase(
            input="Find patient Jane Doe",
            actual_output=(
                "Found Jane Doe, patient 2001, age 42, diagnosed with "
                "Diabetes and admitted on 2024-03-10."
            ),
        )

        metric = measure_answer_relevance(
            test_case, monkeypatch, Verdict.YES
        )

        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_answer_relevance_billing_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pass a relevant answer containing the requested billing details."""
        test_case = LLMTestCase(
            input="Show billing details and length of stay for patient 12345",
            actual_output=(
                "Patient 12345 was billed $12,500.50 and stayed for 10 days."
            ),
        )

        metric = measure_answer_relevance(
            test_case, monkeypatch, Verdict.YES
        )

        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_irrelevant_answer_fails_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail an answer relevance case judged unrelated to its question."""
        test_case = LLMTestCase(
            input="Show billing details for patient 12345",
            actual_output="The weather is sunny today.",
        )

        metric = measure_answer_relevance(
            test_case, monkeypatch, Verdict.NO
        )

        assert metric.score < METRIC_THRESHOLD
        assert metric.success is False


class TestAgentFaithfulness:
    """Evaluate whether responses are grounded in retrieved healthcare data."""

    @pytest.mark.unit
    @pytest.mark.agent
    def test_faithfulness_no_hallucination(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pass a patient summary whose claims are grounded in retrieved data."""
        test_case = LLMTestCase(
            input="Find patient Robert Brown",
            actual_output=(
                "Robert Brown (ID 1003) is 62, has Cancer, and was treated "
                "at Oncology Institute from 2024-03-05 to 2024-03-20."
            ),
            retrieval_context=[
                (
                    "ID 1003, Robert Brown, age 62, Cancer, Oncology Institute, "
                    "admitted 2024-03-05, discharged 2024-03-20."
                )
            ],
        )

        metric = measure_faithfulness(
            test_case, monkeypatch, [Verdict.YES]
        )

        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_faithfulness_billing_accuracy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pass billing details and length of stay grounded in retrieved data."""
        test_case = LLMTestCase(
            input="Get billing summary for patient 1001",
            actual_output=(
                "John Smith (ID 1001) was billed $12,500.50 and stayed "
                "for 10 days from 2024-01-15 to 2024-01-25."
            ),
            retrieval_context=[
                (
                    "Patient 1001: John Smith, billing amount 12500.50, "
                    "admission 2024-01-15, discharge 2024-01-25."
                )
            ],
        )

        metric = measure_faithfulness(
            test_case, monkeypatch, [Verdict.YES]
        )

        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_hallucinated_claim_fails_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail a faithfulness case containing an unsupported claim."""
        test_case = LLMTestCase(
            input="Find patient Robert Brown",
            actual_output=(
                "Robert Brown has Cancer and was treated by a psychiatrist."
            ),
            retrieval_context=[
                "Robert Brown has Cancer and was treated at Oncology Institute."
            ],
        )

        metric = measure_faithfulness(
            test_case, monkeypatch, [Verdict.YES, Verdict.NO]
        )

        assert metric.score < METRIC_THRESHOLD
        assert metric.success is False


class TestAgentEndToEndEvaluation:
    """Evaluate combined tool and answer quality for complete agent workflows."""

    @pytest.mark.unit
    @pytest.mark.agent
    def test_e2e_find_and_get_billing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Score the two expected tool calls and the resulting billing answer."""
        test_case = LLMTestCase(
            input="Find John Smith and show their billing summary",
            actual_output=(
                "John Smith (ID 1001) was billed $12,500.50 for a 10-day "
                "stay at City General Hospital with BlueCross insurance."
            ),
            retrieval_context=[
                "John Smith, ID 1001.",
                (
                    "Patient 1001, bill 12500.50, 10-day stay at City General "
                    "Hospital, BlueCross insurance."
                ),
            ],
            tools_called=[
                ToolCall(
                    name="search_patient_by_name",
                    input_parameters={
                        "first_name": "John",
                        "last_name": "Smith",
                    },
                ),
                ToolCall(
                    name="get_patient_billing_summary",
                    input_parameters={"patient_id": 1001},
                ),
            ],
            expected_tools=[
                ToolCall(
                    name="search_patient_by_name",
                    input_parameters={
                        "first_name": "John",
                        "last_name": "Smith",
                    },
                ),
                ToolCall(
                    name="get_patient_billing_summary",
                    input_parameters={"patient_id": 1001},
                ),
            ],
        )

        tool_metric = measure_tool_correctness(test_case)
        relevance_metric = measure_answer_relevance(
            test_case, monkeypatch, Verdict.YES
        )
        faithfulness_metric = measure_faithfulness(
            test_case, monkeypatch, [Verdict.YES]
        )

        assert tool_metric.success is True
        assert relevance_metric.success is True
        assert faithfulness_metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_e2e_condition_population_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Score condition-query tool selection and its grounded answer."""
        test_case = LLMTestCase(
            input="Show me patients with Hypertension",
            actual_output=(
                "Jane Doe (ID 1002) is a 38-year-old patient with "
                "Hypertension at Metro Medical Center."
            ),
            retrieval_context=[
                (
                    "Patient 1002, Jane Doe, age 38, Hypertension, "
                    "Metro Medical Center."
                )
            ],
            tools_called=[
                ToolCall(
                    name="query_patients_by_condition",
                    input_parameters={
                        "medical_condition": "Hypertension",
                        "limit": 5,
                    },
                )
            ],
            expected_tools=[
                ToolCall(
                    name="query_patients_by_condition",
                    input_parameters={
                        "medical_condition": "Hypertension",
                        "limit": 5,
                    },
                )
            ],
        )

        tool_metric = measure_tool_correctness(test_case)
        relevance_metric = measure_answer_relevance(
            test_case, monkeypatch, Verdict.YES
        )
        faithfulness_metric = measure_faithfulness(
            test_case, monkeypatch, [Verdict.YES]
        )

        assert tool_metric.success is True
        assert relevance_metric.success is True
        assert faithfulness_metric.success is True


class TestAgentConfiguration:
    """Test agent configuration and setup."""

    @pytest.mark.unit
    def test_llm_configuration_values(self) -> None:
        """Verify the configured Unsloth endpoint, API key, and model."""
        assert APP_CONFIG.llm.base_url == "http://localhost:8888/v1", (
            "Base URL should match Unsloth endpoint"
        )
        assert (
            APP_CONFIG.llm.api_key
            == "sk-unsloth-8d52862f0bc3772c0dde526cab00cd1c"
        ), "API key should match specified key"
        assert (
            APP_CONFIG.llm.model == "unsloth/gemma-3-4b-it-GGUF:BF16"
        ), "Model should match specified model"

    @pytest.mark.unit
    def test_database_configuration_default(self) -> None:
        """Verify database configuration contains a PostgreSQL URL."""
        assert "postgresql://" in APP_CONFIG.database.database_url
        assert "healthcare" in APP_CONFIG.database.database_url

    @pytest.mark.unit
    @patch("healthcare_agent.agent.ChatOpenAI")
    def test_agent_initialization(self, mock_chat_openai: Mock) -> None:
        """Verify agent initialization passes the configured LLM settings."""
        mock_chat_openai.return_value = MagicMock()

        agent = HealthcareAgent()

        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:8888/v1"
        assert (
            call_kwargs["api_key"]
            == "sk-unsloth-8d52862f0bc3772c0dde526cab00cd1c"
        )
        assert call_kwargs["model"] == "unsloth/gemma-3-4b-it-GGUF:BF16"
        assert len(agent.tools) == 3

    @pytest.mark.unit
    def test_agent_has_correct_tools(self) -> None:
        """Verify agent exposes the expected healthcare query tools."""
        from healthcare_agent.tools import AVAILABLE_TOOLS

        tool_names = [tool.name for tool in AVAILABLE_TOOLS]

        assert "search_patient_by_name" in tool_names
        assert "get_patient_billing_summary" in tool_names
        assert "query_patients_by_condition" in tool_names
        assert len(tool_names) == 3
