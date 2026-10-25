"""
Repository Module for Healthcare Records Data Access.

This module implements the data access layer (Repository Pattern) for the
healthcare_records table. It provides reusable, type-safe methods for
querying patient data with comprehensive error handling and logging-ready
structure.

All database operations are performed through SQLAlchemy sessions with
proper transaction management. Results are returned as structured dictionaries
for easy consumption by the agent tools layer.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .database import db_manager


class HealthcareRepository:
    """
    Repository class for healthcare_records table operations.
    
    This class encapsulates all SQL queries for patient data access,
    providing a clean interface between the database and business logic.
    It follows the Repository Pattern to isolate data access concerns.
    
    All methods include comprehensive error handling and return structured
    data types suitable for JSON serialization and agent consumption.
    """
    
    def search_patient_by_name(self, first_name: str, last_name: str) -> List[Dict[str, Any]]:
        """
        Search for patients by first name and last name.
        
        Queries the healthcare_records table using case-insensitive matching
        on name and surname columns. Returns all matching patient profiles
        with complete record details.
        
        Args:
            first_name (str): Patient's first name (name column).
            last_name (str): Patient's last name (surname column).
            
        Returns:
            List[Dict[str, Any]]: List of matching patient records as dictionaries.
                                  Each dict contains all columns from healthcare_records.
                                  Returns empty list if no matches found.
                                  
        Raises:
            SQLAlchemyError: If database query fails.
            ValueError: If first_name or last_name are empty or whitespace-only.
            
        Note:
            Uses ILIKE for case-insensitive partial matching to improve usability.
        """
        # Input validation
        if not first_name or not first_name.strip():
            raise ValueError("first_name cannot be empty or whitespace-only")
        
        if not last_name or not last_name.strip():
            raise ValueError("last_name cannot be empty or whitespace-only")
        
        # Normalize inputs (strip whitespace)
        normalized_first = first_name.strip()
        normalized_last = last_name.strip()
        
        try:
            # Execute query with case-insensitive matching
            with db_manager.get_session() as session:
                # SQL query using ILIKE for flexible name matching
                # Orders results by id for consistent output
                query = text("""
                    SELECT 
                        id,
                        name,
                        surname,
                        age,
                        gender,
                        blood_type,
                        medical_condition,
                        date_of_admission,
                        doctor,
                        hospital,
                        insurance_provider,
                        billing_amount,
                        room_number,
                        admission_type,
                        discharge_date,
                        medication,
                        test_results
                    FROM healthcare_records
                    WHERE LOWER(TRIM(name)) = LOWER(TRIM(:first_name))
                      AND LOWER(TRIM(surname)) = LOWER(TRIM(:last_name))
                    ORDER BY id ASC
                """)
                
                # Execute with parameterized query to prevent SQL injection
                result = session.execute(
                    query,
                    {
                        "first_name": normalized_first,
                        "last_name": normalized_last
                    }
                )
                
                # Fetch all matching rows
                rows = result.fetchall()
                
                # Convert rows to list of dictionaries
                patients: List[Dict[str, Any]] = []
                for row in rows:
                    # Map row to dictionary with column names
                    patient_dict = {
                        "id": row.id,
                        "name": row.name,
                        "surname": row.surname,
                        "age": row.age,
                        "gender": row.gender,
                        "blood_type": row.blood_type,
                        "medical_condition": row.medical_condition,
                        "date_of_admission": row.date_of_admission,
                        "doctor": row.doctor,
                        "hospital": row.hospital,
                        "insurance_provider": row.insurance_provider,
                        "billing_amount": float(row.billing_amount) if row.billing_amount is not None else None,
                        "room_number": row.room_number,
                        "admission_type": row.admission_type,
                        "discharge_date": row.discharge_date,
                        "medication": row.medication,
                        "test_results": row.test_results
                    }
                    patients.append(patient_dict)
                
                return patients
                
        except SQLAlchemyError as db_error:
            # Log and re-raise database errors
            raise SQLAlchemyError(f"Failed to search patients by name: {str(db_error)}") from db_error
    
    def get_patient_billing_summary(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve billing cost, hospital, insurance provider, and length of stay by patient ID.
        
        Queries the healthcare_records table for a specific patient and calculates
        the length of stay in days based on admission and discharge dates.
        Returns None if patient not found.
        
        Args:
            patient_id (int): Primary key of the patient record (id column).
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing billing summary with keys:
                                      - patient_id: int
                                      - name: str
                                      - surname: str
                                      - hospital: str
                                      - insurance_provider: str
                                      - billing_amount: float
                                      - date_of_admission: date
                                      - discharge_date: date
                                      - length_of_stay_days: int (calculated)
                                      Returns None if patient not found.
                                      
        Raises:
            SQLAlchemyError: If database query fails.
            ValueError: If patient_id is not a positive integer.
        """
        # Input validation
        if not isinstance(patient_id, int) or patient_id <= 0:
            raise ValueError("patient_id must be a positive integer")
        
        try:
            # Execute query for specific patient
            with db_manager.get_session() as session:
                query = text("""
                    SELECT 
                        id,
                        name,
                        surname,
                        hospital,
                        insurance_provider,
                        billing_amount,
                        date_of_admission,
                        discharge_date
                    FROM healthcare_records
                    WHERE id = :patient_id
                """)
                
                # Execute with parameterized query
                result = session.execute(query, {"patient_id": patient_id})
                row = result.fetchone()
                
                # Return None if patient not found
                if row is None:
                    return None
                
                # Calculate length of stay in days
                length_of_stay_days: int = 0
                if row.date_of_admission is not None and row.discharge_date is not None:
                    # Calculate difference between discharge and admission
                    stay_delta = row.discharge_date - row.date_of_admission
                    length_of_stay_days = stay_delta.days
                    # Ensure non-negative (data should be clean, but defensive)
                    if length_of_stay_days < 0:
                        length_of_stay_days = 0
                
                # Build billing summary dictionary
                billing_summary: Dict[str, Any] = {
                    "patient_id": row.id,
                    "name": row.name,
                    "surname": row.surname,
                    "hospital": row.hospital,
                    "insurance_provider": row.insurance_provider,
                    "billing_amount": float(row.billing_amount) if row.billing_amount is not None else None,
                    "date_of_admission": row.date_of_admission,
                    "discharge_date": row.discharge_date,
                    "length_of_stay_days": length_of_stay_days
                }
                
                return billing_summary
                
        except SQLAlchemyError as db_error:
            # Log and re-raise database errors
            raise SQLAlchemyError(f"Failed to retrieve billing summary for patient {patient_id}: {str(db_error)}") from db_error
    
    def query_patients_by_condition(self, medical_condition: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Filter patients by medical condition and return a summary.
        
        Queries the healthcare_records table for patients with the specified
        medical condition. Returns a summary view with key demographic and
        admission details, limited by the specified count.
        
        Args:
            medical_condition (str): Medical condition to filter by 
                                    (e.g., 'Diabetes', 'Cancer', 'Hypertension', 
                                     'Arthritis', 'Obesity', 'Asthma').
            limit (int): Maximum number of records to return. Defaults to 5.
                        Must be between 1 and 100 for safety.
            
        Returns:
            List[Dict[str, Any]]: List of patient summary dictionaries.
                                  Each contains: id, name, surname, age, gender,
                                  medical_condition, hospital, admission_type, doctor.
                                  Returns empty list if no matches found.
                                  
        Raises:
            SQLAlchemyError: If database query fails.
            ValueError: If medical_condition is empty or limit is out of range.
        """
        # Input validation
        if not medical_condition or not medical_condition.strip():
            raise ValueError("medical_condition cannot be empty or whitespace-only")
        
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValueError("limit must be an integer between 1 and 100")
        
        # Normalize input
        normalized_condition = medical_condition.strip()
        
        try:
            # Execute query filtered by medical condition
            with db_manager.get_session() as session:
                query = text("""
                    SELECT 
                        id,
                        name,
                        surname,
                        age,
                        gender,
                        medical_condition,
                        hospital,
                        admission_type,
                        doctor,
                        date_of_admission
                    FROM healthcare_records
                    WHERE LOWER(TRIM(medical_condition)) = LOWER(TRIM(:medical_condition))
                    ORDER BY id ASC
                    LIMIT :limit
                """)
                
                # Execute with parameterized query
                result = session.execute(
                    query,
                    {
                        "medical_condition": normalized_condition,
                        "limit": limit
                    }
                )
                
                # Fetch all matching rows
                rows = result.fetchall()
                
                # Convert rows to list of summary dictionaries
                patient_summaries: List[Dict[str, Any]] = []
                for row in rows:
                    summary_dict = {
                        "id": row.id,
                        "name": row.name,
                        "surname": row.surname,
                        "age": row.age,
                        "gender": row.gender,
                        "medical_condition": row.medical_condition,
                        "hospital": row.hospital,
                        "admission_type": row.admission_type,
                        "doctor": row.doctor,
                        "date_of_admission": row.date_of_admission
                    }
                    patient_summaries.append(summary_dict)
                
                return patient_summaries
                
        except SQLAlchemyError as db_error:
            # Log and re-raise database errors
            raise SQLAlchemyError(f"Failed to query patients by condition '{medical_condition}': {str(db_error)}") from db_error


# Global repository instance
# Shared across the application for consistent data access
healthcare_repository = HealthcareRepository()
