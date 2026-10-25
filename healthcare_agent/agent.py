"""
Agent Module for Healthcare AI Agent with Tool-Calling.

This module handles agent initialization, tool binding, and execution.
It uses LangChain's OpenAI-compatible ChatOpenAI client to connect to
the Unsloth endpoint and implements a manual ReAct-style agent loop
for models that do not support native function/tool calling.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import APP_CONFIG
from .database import db_manager
from .tools import AVAILABLE_TOOLS


class HealthcareAgent:
    """
    Healthcare Data Agent with manual tool-calling capabilities.
    """
    
    def __init__(self) -> None:
        """Initialize HealthcareAgent with LLM and tools."""
        self.llm = ChatOpenAI(
            base_url=APP_CONFIG.llm.base_url,
            api_key=APP_CONFIG.llm.api_key,
            model=APP_CONFIG.llm.model,
            temperature=0.0,
            max_tokens=512,
            timeout=60,
            max_retries=1
        )
        self.tools = AVAILABLE_TOOLS
        self.tool_map: Dict[str, Any] = {tool.name: tool for tool in self.tools}
    
    def query(self, user_question: str) -> Dict[str, Any]:
        """
        Process a user question and return agent response.
        """
        if not user_question or not user_question.strip():
            return {
                "success": False,
                "answer": "Error: User question cannot be empty.",
                "error": "Empty question provided"
            }
        
        try:
            if not db_manager.test_connection():
                return {
                    "success": False,
                    "answer": "Error: Unable to connect to the database.",
                    "error": "Database connection failed"
                }
            
            # Simple direct approach
            q = user_question.lower()
            
            # Try to be smart about tool usage
            system_msg = SystemMessage(content="""You are a helpful healthcare data assistant. 
You have access to these tools:
1. search_patient_by_name(first_name, last_name) - find patients by name
2. get_patient_billing_summary(patient_id) - get billing info
3. query_patients_by_condition(medical_condition, limit) - find patients by condition

To use a tool, respond EXACTLY like this:
TOOL: tool_name
ARGS: {"param": "value"}

After getting results, give a clear answer. Be concise.""")
            
            messages = [system_msg, HumanMessage(content=user_question)]
            
            for i in range(3):  # Max 3 iterations
                resp = self.llm.invoke(messages)
                content = resp.content
                
                # Check for tool call
                tool_match = re.search(r'(?i)TOOL:\s*(\w+)', content)
                if tool_match:
                    tool_name = tool_match.group(1)
                    args_match = re.search(r'(?i)ARGS:\s*(\{[^{}]*\})', content)
                    if args_match and tool_name in self.tool_map:
                        try:
                            args = json.loads(args_match.group(1))
                            tool_result = self.tool_map[tool_name].invoke(args)
                            messages.append(AIMessage(content=content))
                            messages.append(HumanMessage(content=f"Tool result: {tool_result}\n\nAnswer in plain English now."))
                            continue
                        except Exception as e:
                            messages.append(HumanMessage(content=f"Error: {e}. Try again or answer directly."))
                            continue
                
                # No tool call, return answer
                answer = re.sub(r'(?i)^TOOL:.*$', '', content, flags=re.MULTILINE)
                answer = re.sub(r'(?i)^ARGS:.*$', '', answer, flags=re.MULTILINE)
                answer = answer.strip()
                if not answer:
                    answer = content.strip()
                return {"success": True, "answer": answer}
            
            return {"success": False, "answer": "Max iterations reached", "error": "timeout"}
            
        except Exception as e:
            return {"success": False, "answer": f"Error: {str(e)}", "error": str(e)}
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        tool_info = []
        for tool in self.tools:
            tool_info.append({"name": tool.name, "description": tool.description.split('\n')[0]})
        return tool_info


healthcare_agent = HealthcareAgent()
