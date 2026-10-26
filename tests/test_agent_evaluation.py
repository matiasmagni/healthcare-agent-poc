"""Live DeepEval evaluations of the healthcare agent using a local Unsloth judge."""

from typing import Any, Optional, Type
from unittest.mock import MagicMock, Mock, patch

from openai import AsyncOpenAI, OpenAI
import pytest
from pydantic import BaseModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ToolCorrectnessMetric,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

from healthcare_agent.agent import HealthcareAgent
from healthcare_agent.config import APP_CONFIG

METRIC_THRESHOLD = 0.7


class UnslothDeepEvalJudge(DeepEvalBaseLLM):
    """DeepEval judge adapter backed by the local OpenAI-compatible Unsloth API."""

    def __init__(self) -> None:
        """Initialize DeepEval with the configured Unsloth model name."""
        self.unsloth_model = APP_CONFIG.llm.model
        super().__init__(model=self.unsloth_model)

    def load_model(self, *args: Any, **kwargs: Any) -> "UnslothDeepEvalJudge":
        """Create sync and async OpenAI clients for the local judge endpoint."""
        self.client: OpenAI = OpenAI(
            base_url=APP_CONFIG.llm.base_url,
            api_key=APP_CONFIG.llm.api_key,
        )
        self.async_client: AsyncOpenAI = AsyncOpenAI(
            base_url=APP_CONFIG.llm.base_url,
            api_key=APP_CONFIG.llm.api_key,
        )
        return self

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a metric judgment using the synchronous chat completions API."""
        request: dict[str, Any] = {
            "model": self.unsloth_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if schema is not None:
            request["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Unsloth returned an empty evaluation response")
        return content

    async def async_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a metric judgment using the async chat completions API."""
        request: dict[str, Any] = {
            "model": self.unsloth_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if schema is not None:
            request["response_format"] = {"type": "json_object"}
        response = await self.async_client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Unsloth returned an empty evaluation response")
        return content

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        """Implement DeepEval's async generation interface."""
        return await self.async_generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        """Return the model identifier used for evaluation reporting."""
        return self.unsloth_model


UNSLOTH_JUDGE = UnslothDeepEvalJudge()
AVAILABLE_TOOL_CALLS = [
    ToolCall(
        name="search_patient_by_name",
        description="Find patients using their first and last names.",
        input_parameters={},
    ),
    ToolCall(
        name="get_patient_billing_summary",
        description="Retrieve billing details using a patient ID.",
        input_parameters={},
    ),
    ToolCall(
        name="query_patients_by_condition",
        description="Find patients with a medical condition.",
        input_parameters={},
    ),
]


def print_metric_report(
    test_case: LLMTestCase,
    metric: BaseMetric,
    is_negative_test: bool = False,
) -> None:
    """Print the metric score, threshold, outcome, and live judge explanation."""
    if is_negative_test:
        # Negative tests intentionally expect the metric to fall below threshold.
        status = (
            "(EXPECTED FAIL - Negative Test Case)"
            if not metric.success
            else "(UNEXPECTED PASS - Negative Test Case)"
        )
    else:
        status = "PASS" if metric.success else "FAIL"
    reason = metric.reason or "No reason returned by the metric."
    print(
        "\n"
        + "=" * 78
        + f"\nDeepEval: {test_case.name or test_case.input}"
        + f"\nMetric: {metric.__name__}"
        + f"\nJudge: {metric.evaluation_model or APP_CONFIG.llm.model}"
        + f"\nScore: {metric.score:.3f} | Threshold: {metric.threshold}"
        + f"\nStatus: {status}"
        + f"\nReason: {reason}"
        + "\n"
        + "=" * 78,
        flush=True,
    )


def measure_tool_correctness(
    test_case: LLMTestCase,
    is_negative_test: bool = False,
) -> ToolCorrectnessMetric:
    """Run live DeepEval tool correctness for selection and argument accuracy."""
    metric = ToolCorrectnessMetric(
        threshold=METRIC_THRESHOLD,
        available_tools=AVAILABLE_TOOL_CALLS,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        async_mode=False,
        should_exact_match=True,
        include_reason=True,
        model=UNSLOTH_JUDGE,
    )
    metric.measure(test_case, _show_indicator=False)
    print_metric_report(test_case, metric, is_negative_test=is_negative_test)
    return metric


def measure_answer_relevance(
    test_case: LLMTestCase,
    is_negative_test: bool = False,
) -> AnswerRelevancyMetric:
    """Measure answer relevance using the live Unsloth LLM judge."""
    metric = AnswerRelevancyMetric(
        threshold=METRIC_THRESHOLD,
        async_mode=False,
        include_reason=True,
        model=UNSLOTH_JUDGE,
    )
    metric.measure(test_case, _show_indicator=False)
    print_metric_report(test_case, metric, is_negative_test=is_negative_test)
    return metric


def measure_faithfulness(
    test_case: LLMTestCase,
    is_negative_test: bool = False,
) -> FaithfulnessMetric:
    """Measure faithfulness using the live Unsloth LLM judge."""
    metric = FaithfulnessMetric(
        threshold=METRIC_THRESHOLD,
        async_mode=False,
        include_reason=True,
        model=UNSLOTH_JUDGE,
    )
    metric.measure(test_case, _show_indicator=False)
    print_metric_report(test_case, metric, is_negative_test=is_negative_test)
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
            name="test_tool_selection_search_by_name",
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

        assert metric.score is not None
        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_tool_selection_billing_summary(self) -> None:
        """Score the expected billing-summary tool call and patient ID."""
        test_case = LLMTestCase(
            input="Get billing summary for patient ID 12345",
            actual_output="Billing summary retrieved.",
            name="test_tool_selection_billing_summary",
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

        assert metric.score is not None
        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_tool_selection_condition_query(self) -> None:
        """Score condition-query tool selection, condition, and requested limit."""
        test_case = LLMTestCase(
            input="Show me 5 patients with Diabetes",
            actual_output="Condition query completed.",
            name="test_tool_selection_condition_query",
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

        assert metric.score is not None
        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_wrong_tool_selection_fails_threshold(self) -> None:
        """Reject an incorrect tool even when a tool was called."""
        test_case = LLMTestCase(
            input="Find patient John Smith",
            actual_output="Billing summary retrieved.",
            name="test_wrong_tool_selection_fails_threshold",
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

        metric = measure_tool_correctness(test_case, is_negative_test=True)

        assert metric.score is not None
        assert metric.score < METRIC_THRESHOLD
        assert metric.success is False


class TestAgentAnswerRelevance:
    """Evaluate whether agent answers address the user's question."""

    @pytest.mark.unit
    @pytest.mark.agent
    def test_answer_relevance_name_search(self) -> None:
        """Pass a relevant answer to a patient search request."""
        test_case = LLMTestCase(
            input="Find patient Jane Doe",
            actual_output=(
                "Found Jane Doe, patient 2001, age 42, diagnosed with "
                "Diabetes and admitted on 2024-03-10."
            ),
            name="test_answer_relevance_name_search",
        )

        metric = measure_answer_relevance(test_case)

        assert metric.score is not None
        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_answer_relevance_billing_query(self) -> None:
        """Pass a relevant answer containing the requested billing details."""
        test_case = LLMTestCase(
            input="Show billing details and length of stay for patient 12345",
            actual_output=(
                "Patient 12345 was billed $12,500.50 and stayed for 10 days."
            ),
            name="test_answer_relevance_billing_query",
        )

        metric = measure_answer_relevance(test_case)

        assert metric.score is not None
        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_irrelevant_answer_fails_threshold(self) -> None:
        """Fail an answer relevance case judged unrelated to its question."""
        test_case = LLMTestCase(
            input="Show billing details for patient 12345",
            actual_output="The weather is sunny today.",
            name="test_irrelevant_answer_fails_threshold",
        )

        metric = measure_answer_relevance(test_case, is_negative_test=True)

        assert metric.score is not None
        assert metric.score < METRIC_THRESHOLD
        assert metric.success is False


class TestAgentFaithfulness:
    """Evaluate whether responses are grounded in retrieved healthcare data."""

    @pytest.mark.unit
    @pytest.mark.agent
    def test_faithfulness_no_hallucination(self) -> None:
        """Pass a patient summary whose claims are grounded in retrieved data."""
        test_case = LLMTestCase(
            input="Find patient Robert Brown",
            actual_output=(
                "Robert Brown (ID 1003) is 62, has Cancer, and was treated "
                "at Oncology Institute from 2024-03-05 to 2024-03-20."
            ),
            name="test_faithfulness_no_hallucination",
            retrieval_context=[
                (
                    "ID 1003, Robert Brown, age 62, Cancer, Oncology Institute, "
                    "admitted 2024-03-05, discharged 2024-03-20."
                )
            ],
        )

        metric = measure_faithfulness(test_case)

        assert metric.score is not None
        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_faithfulness_billing_accuracy(self) -> None:
        """Pass billing amount and admission dates grounded in retrieved data."""
        test_case = LLMTestCase(
            input="Get billing summary for patient 1001",
            actual_output=(
                "John Smith (ID 1001) was billed $12,500.50, admitted "
                "2024-01-15, and discharged 2024-01-25."
            ),
            name="test_faithfulness_billing_accuracy",
            retrieval_context=[
                (
                    "Patient 1001: John Smith, billing amount 12500.50, "
                    "admission 2024-01-15, discharge 2024-01-25."
                )
            ],
        )

        metric = measure_faithfulness(test_case)

        assert metric.score is not None
        assert metric.score >= METRIC_THRESHOLD
        assert metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_hallucinated_claim_fails_threshold(self) -> None:
        """Fail a faithfulness case containing an unsupported claim."""
        test_case = LLMTestCase(
            input="Find patient Robert Brown",
            actual_output=(
                "Robert Brown has Cancer and was treated by a psychiatrist."
            ),
            name="test_hallucinated_claim_fails_threshold",
            retrieval_context=[
                "Robert Brown has Cancer and was treated at Oncology Institute."
            ],
        )

        metric = measure_faithfulness(test_case, is_negative_test=True)

        assert metric.score is not None
        assert metric.score < METRIC_THRESHOLD
        assert metric.success is False


class TestAgentEndToEndEvaluation:
    """Evaluate combined tool and answer quality for complete agent workflows."""

    @pytest.mark.unit
    @pytest.mark.agent
    def test_e2e_find_and_get_billing(self) -> None:
        """Score the two expected tool calls and the resulting billing answer."""
        test_case = LLMTestCase(
            input=(
                "Find John Smith and show the bill, hospital, insurance, "
                "and length of stay."
            ),
            actual_output=(
                "John Smith (ID 1001) was billed $12,500.50 for a 10-day "
                "stay at City General Hospital with BlueCross insurance."
            ),
            name="test_e2e_find_and_get_billing",
            retrieval_context=[
                "John Smith, ID 1001.",
                (
                    "Patient 1001, bill 12500.50, 10-day stay at City General "
                    "Hospital, BlueCross insurance, length of stay 10 days."
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
        relevance_metric = measure_answer_relevance(test_case)
        faithfulness_metric = measure_faithfulness(test_case)

        assert tool_metric.success is True
        assert relevance_metric.success is True
        assert faithfulness_metric.success is True

    @pytest.mark.unit
    @pytest.mark.agent
    def test_e2e_condition_population_query(self) -> None:
        """Score condition-query tool selection and its grounded answer."""
        test_case = LLMTestCase(
            input="Show me patients with Hypertension",
            actual_output=(
                "Jane Doe (ID 1002) is a 38-year-old patient with "
                "Hypertension at Metro Medical Center."
            ),
            name="test_e2e_condition_population_query",
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
        relevance_metric = measure_answer_relevance(test_case)
        faithfulness_metric = measure_faithfulness(test_case)

        assert tool_metric.success is True
        assert relevance_metric.success is True
        assert faithfulness_metric.success is True


class TestAgentConfiguration:
    """Test agent configuration and setup."""

    @pytest.mark.unit
    def test_llm_configuration_values(self) -> None:
        """Verify Unsloth settings are loaded from the environment-backed config."""
        assert APP_CONFIG.llm.base_url
        assert APP_CONFIG.llm.api_key
        assert APP_CONFIG.llm.model

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
        assert call_kwargs["base_url"] == APP_CONFIG.llm.base_url
        assert call_kwargs["api_key"] == APP_CONFIG.llm.api_key
        assert call_kwargs["model"] == APP_CONFIG.llm.model
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
