"""
Tools Module for Healthcare Agent.

This module defines all agent tools that wrap repository functions for
safe database access. Tools are created using LangChain's @tool decorator
with comprehensive type hints, descriptions, and error handling.

Each tool:
- Validates inputs
- Calls the appropriate repository method
- Formats results for agent consumption
- Handles errors gracefully with informative messages
- Returns structured, human-readable output

Tools are bound to the agent to enable tool-calling functionality.
"""

import json
from typing import Type

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from .repository import healthcare_repository


# -------------------- Tool Input Schemas --------------------

class SearchPatientByNameInput(BaseModel):
    """
    Input schema for search_patient_by_name tool.
    
    Defines validated parameters for patient name search.
    """
    first_name: str = Field(
        description="Patient's first name (e.g., 'John', 'Jane')"
    )
    last_name: str = Field(
        description="Patient's last name (e.g., 'Smith', 'Doe')"
    )


class GetPatientBillingSummaryInput(BaseModel):
    """
    Input schema for get_patient_billing_summary tool.
    
    Defines validated parameters for billing summary retrieval.
    """
    patient_id: int = Field(
        description="Patient's unique identifier (primary key from healthcare_records)",
        gt=0
    )


class QueryPatientsByConditionInput(BaseModel):
    """
    Input schema for query_patients_by_condition tool.
    
    Defines validated parameters for condition-based patient queries.
    """
    medical_condition: str = Field(
        description="Medical condition to filter by (e.g., 'Diabetes', 'Cancer', 'Hypertension', 'Arthritis', 'Obesity', 'Asthma')"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of results to return (1-100)",
        ge=1,
        le=100
    )


# -------------------- Tool Definitions --------------------

@tool(args_schema=SearchPatientByNameInput)
def search_patient_by_name(first_name: str, last_name: str) -> str:
    """
    Search for patients by first name and last name in the healthcare_records database.
    
    This tool queries the PostgreSQL database using case-insensitive matching to find
    patient records. It returns complete patient profiles including demographics, 
    medical condition, admission details, doctor, hospital, and all other record fields.
    
    Use this tool when:
    - User wants to find a specific patient by name
    - User needs complete patient profile information
    - Need to locate patient before retrieving billing details
    
    Args:
        first_name (str): Patient's first name
        last_name (str): Patient's last name
        
    Returns:
        str: JSON-formatted string containing search results. Returns empty array
             message if no patients found.
    """
    try:
        # Call repository to search for patients
        patients = healthcare_repository.search_patient_by_name(
            first_name=first_name,
            last_name=last_name
        )
        
        # Handle case where no patients found
        if not patients:
            return json.dumps({
                "status": "no_results",
                "message": f"No patients found with name '{first_name} {last_name}'",
                "results": []
            }, indent=2, default=str)
        
        # Return results as formatted JSON
        return json.dumps({
            "status": "success",
            "count": len(patients),
            "message": f"Found {len(patients)} patient(s) matching '{first_name} {last_name}'",
            "results": patients
        }, indent=2, default=str)
        
    except ValueError as val_error:
        # Handle input validation errors
        return json.dumps({
            "status": "error",
            "error_type": "validation_error",
            "message": str(val_error)
        }, indent=2)
        
    except SQLAlchemyError as db_error:
        # Handle database errors
        return json.dumps({
            "status": "error",
            "error_type": "database_error",
            "message": f"Database error occurred while searching for patients: {str(db_error)}"
        }, indent=2)
        
    except Exception as general_error:
        # Handle unexpected errors
        return json.dumps({
            "status": "error",
            "error_type": "unknown_error",
            "message": f"Unexpected error occurred: {str(general_error)}"
        }, indent=2)


@tool(args_schema=GetPatientBillingSummaryInput)
def get_patient_billing_summary(patient_id: int) -> str:
    """
    Retrieve billing cost, hospital, insurance provider, and length of stay by patient ID.
    
    This tool fetches billing and admission details for a specific patient and
    automatically calculates the length of stay in days from admission and discharge dates.
    
    Use this tool when:
    - User wants billing information for a specific patient
    - Need to know hospital, insurance provider, or costs
    - Need to calculate or retrieve length of stay
    - Already have patient_id from previous search
    
    Args:
        patient_id (int): Patient's unique identifier (id from healthcare_records)
        
    Returns:
        str: JSON-formatted string containing billing summary. Returns error
             if patient not found or other issues occur.
    """
    try:
        # Call repository to get billing summary
        billing_summary = healthcare_repository.get_patient_billing_summary(
            patient_id=patient_id
        )
        
        # Handle case where patient not found
        if billing_summary is None:
            return json.dumps({
                "status": "no_results",
                "message": f"No patient found with patient_id {patient_id}",
                "result": None
            }, indent=2)
        
        # Return results as formatted JSON
        return json.dumps({
            "status": "success",
            "message": f"Billing summary retrieved for patient {patient_id}",
            "result": billing_summary
        }, indent=2, default=str)
        
    except ValueError as val_error:
        # Handle input validation errors
        return json.dumps({
            "status": "error",
            "error_type": "validation_error",
            "message": str(val_error)
        }, indent=2)
        
    except SQLAlchemyError as db_error:
        # Handle database errors
        return json.dumps({
            "status": "error",
            "error_type": "database_error",
            "message": f"Database error occurred while retrieving billing summary: {str(db_error)}"
        }, indent=2)
        
    except Exception as general_error:
        # Handle unexpected errors
        return json.dumps({
            "status": "error",
            "error_type": "unknown_error",
            "message": f"Unexpected error occurred: {str(general_error)}"
        }, indent=2)


@tool(args_schema=QueryPatientsByConditionInput)
def query_patients_by_condition(medical_condition: str, limit: int = 5) -> str:
    """
    Filter patients by medical condition and return a summary.
    
    This tool queries the database for patients diagnosed with a specific medical
    condition. It returns a summary view with key information including demographics,
    hospital, admission type, and doctor. Results are limited for readability.
    
    Supported conditions (based on dataset): Arthritis, Diabetes, Hypertension, Obesity, Cancer, Asthma
    
    Use this tool when:
    - User wants to find patients with a specific condition
    - Need population overview for a medical condition
    - Researching patients by diagnosis
    - User doesn't know specific patient names
    
    Args:
        medical_condition (str): Medical condition to filter by
        limit (int): Maximum number of results to return (default: 5, max: 100)
        
    Returns:
        str: JSON-formatted string containing patient summaries. Returns empty
             array message if no patients found.
    """
    try:
        # Call repository to query by condition
        patient_summaries = healthcare_repository.query_patients_by_condition(
            medical_condition=medical_condition,
            limit=limit
        )
        
        # Handle case where no patients found
        if not patient_summaries:
            return json.dumps({
                "status": "no_results",
                "message": f"No patients found with medical condition '{medical_condition}'",
                "count": 0,
                "results": []
            }, indent=2, default=str)
        
        # Return results as formatted JSON
        return json.dumps({
            "status": "success",
            "count": len(patient_summaries),
            "message": f"Found {len(patient_summaries)} patient(s) with condition '{medical_condition}' (limit: {limit})",
            "results": patient_summaries
        }, indent=2, default=str)
        
    except ValueError as val_error:
        # Handle input validation errors
        return json.dumps({
            "status": "error",
            "error_type": "validation_error",
            "message": str(val_error)
        }, indent=2)
        
    except SQLAlchemyError as db_error:
        # Handle database errors
        return json.dumps({
            "status": "error",
            "error_type": "database_error",
            "message": f"Database error occurred while querying by condition: {str(db_error)}"
        }, indent=2)
        
    except Exception as general_error:
        # Handle unexpected errors
        return json.dumps({
            "status": "error",
            "error_type": "unknown_error",
            "message": f"Unexpected error occurred: {str(general_error)}"
        }, indent=2)


# List of all available tools for agent binding
AVAILABLE_TOOLS = [
    search_patient_by_name,
    get_patient_billing_summary,
    query_patients_by_condition
]
