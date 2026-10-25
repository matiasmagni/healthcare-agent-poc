"""
Main Entry Point for Healthcare AI Agent.

This module provides an interactive command-line interface for users
to query the healthcare records database through the AI agent.
It handles initialization, user input, and displays formatted results.

Usage:
    python main.py
    
The agent will start an interactive session where users can ask
questions about patients, billing, and medical conditions.
"""

import sys
from typing import Optional

from .agent import healthcare_agent
from .config import APP_CONFIG
from .database import db_manager


def print_banner() -> None:
    """
    Print application banner with configuration information.
    
    Displays agent details, LLM endpoint, and available tools
    for user awareness.
    """
    print("=" * 70)
    print("HEALTHCARE AI AGENT - Tool-Calling with PostgreSQL")
    print("=" * 70)
    print(f"LLM Model: {APP_CONFIG.llm.model}")
    print(f"LLM Endpoint: {APP_CONFIG.llm.base_url}")
    print(f"Database: {APP_CONFIG.database.database_url.split('@')[-1]}")  # Hide credentials
    print(f"Max Iterations: {APP_CONFIG.agent.max_iterations}")
    print(f"Verbose Mode: {APP_CONFIG.agent.verbose}")
    print("=" * 70)
    print("\nAvailable Tools:")
    for i, tool_info in enumerate(healthcare_agent.get_available_tools(), 1):
        print(f"  {i}. {tool_info['name']}: {tool_info['description']}")
    print("\n" + "=" * 70)
    print("Type 'help' for examples, 'tools' for tool details, 'exit' or 'quit' to quit")
    print("=" * 70 + "\n")


def print_help() -> None:
    """
    Print usage examples and help information.
    
    Provides sample queries to help users understand how to
    interact with the agent effectively.
    """
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    print("\n1. Search patient by name:")
    print("   'Find patient John Smith'")
    print("   'Search for Jane Doe'")
    print("\n2. Get billing summary:")
    print("   'Get billing summary for patient ID 12345'")
    print("   'Show me billing details and length of stay for patient 67890'")
    print("\n3. Query by medical condition:")
    print("   'Show me 5 patients with Diabetes'")
    print("   'Find patients with Cancer'")
    print("   'List 10 patients diagnosed with Hypertension'")
    print("\n4. Combined queries:")
    print("   'Find John Smith and show their billing summary'")
    print("   'Get details for all patients with Asthma'")
    print("\n" + "=" * 70 + "\n")


def print_tools_detail() -> None:
    """
    Print detailed information about available tools.
    
    Shows complete descriptions and parameters for each tool
    to help users understand agent capabilities.
    """
    print("\n" + "=" * 70)
    print("TOOL DETAILS")
    print("=" * 70)
    
    # Tool 1: search_patient_by_name
    print("\n1. search_patient_by_name(first_name: str, last_name: str)")
    print("   Description: Search for patients by first and last name")
    print("   Returns: Complete patient profiles with all fields")
    print("   Use case: Find specific patients, get full record details")
    
    # Tool 2: get_patient_billing_summary
    print("\n2. get_patient_billing_summary(patient_id: int)")
    print("   Description: Get billing cost, hospital, insurance, and length of stay")
    print("   Returns: Billing summary with calculated length of stay in days")
    print("   Use case: Financial/administrative queries, hospital stay duration")
    
    # Tool 3: query_patients_by_condition
    print("\n3. query_patients_by_condition(medical_condition: str, limit: int = 5)")
    print("   Description: Filter patients by medical condition")
    print("   Returns: Patient summaries (id, name, age, gender, hospital, doctor)")
    print("   Use case: Population queries, condition-based research")
    print("   Supported: Arthritis, Diabetes, Hypertension, Obesity, Cancer, Asthma")
    
    print("\n" + "=" * 70 + "\n")


def check_database_connection() -> bool:
    """
    Check and report database connection status.
    
    Tests connectivity to PostgreSQL and provides user feedback.
    
    Returns:
        bool: True if database is accessible, False otherwise.
    """
    print("Checking database connection...", end=" ")
    if db_manager.test_connection():
        print("✓ Connected")
        return True
    else:
        print("✗ Failed")
        print("\nError: Could not connect to PostgreSQL database.")
        print("Please ensure:")
        print("  1. PostgreSQL is running")
        print("  2. Database 'healthcare' exists")
        print("  3. Credentials are correct in DATABASE_URL")
        print("  4. Table 'healthcare_records' exists with data")
        return False


def main() -> None:
    """
    Main interactive loop for healthcare agent.
    
    Handles user input, processes queries through the agent,
    and displays results in a formatted manner.
    """
    try:
        # Print banner and check connections
        print_banner()
        
        if not check_database_connection():
            print("\nExiting due to database connection failure.")
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("Agent ready. Enter your query below.")
        print("=" * 70 + "\n")
        
        # Interactive loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle special commands
                command_lower = user_input.lower()
                
                if command_lower in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                elif command_lower == 'help':
                    print_help()
                    continue
                
                elif command_lower == 'tools':
                    print_tools_detail()
                    continue
                
                elif command_lower == 'clear':
                    # Clear screen (Unix/Linux)
                    print("\033[2J\033[H", end="")
                    print_banner()
                    print("\nAgent ready.\n")
                    continue
                
                # Process query through agent
                print("\nAgent: Thinking...", flush=True)
                response = healthcare_agent.query(user_input)
                
                # Display response
                print("\n" + "-" * 70)
                if response.get("success"):
                    print("Agent Response:")
                    print("-" * 70)
                    print(response.get("answer", "No response"))
                    
                    # Display verbose information if enabled
                    if APP_CONFIG.agent.verbose and "messages" in response:
                        print("\n" + "-" * 70)
                        print("Execution Trace (Verbose):")
                        print("-" * 70)
                        for i, msg in enumerate(response["messages"], 1):
                            print(f"\n[{i}] {msg['type']}")
                            if "tool_calls" in msg:
                                for tc in msg["tool_calls"]:
                                    print(f"    Tool Call: {tc['name']}")
                                    print(f"    Args: {tc['args']}")
                            if "tool_name" in msg:
                                print(f"    Tool: {msg['tool_name']}")
                            if "content" in msg and msg["content"]:
                                content = msg["content"]
                                if isinstance(content, str):
                                    # Truncate long content for readability
                                    if len(content) > 500:
                                        print(f"    Content: {content[:500]}... [truncated]")
                                    else:
                                        print(f"    Content: {content}")
                else:
                    print("Agent Error:")
                    print("-" * 70)
                    print(response.get("answer", "Unknown error occurred"))
                    if "error" in response:
                        print(f"\nDetails: {response['error']}")
                
                print("\n" + "=" * 70 + "\n")
                
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\n\nInterrupted. Goodbye!")
                break
            
            except EOFError:
                # Handle end of file
                print("\n\nGoodbye!")
                break
        
    except Exception as fatal_error:
        # Handle fatal errors during startup
        print(f"\nFatal error: {str(fatal_error)}", file=sys.stderr)
        sys.exit(1)
    
    finally:
        # Cleanup database connections on exit
        db_manager.dispose()


# Run the main function
if __name__ == "__main__":
    main()
