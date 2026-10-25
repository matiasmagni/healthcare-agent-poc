"""
Database Module for Healthcare Agent.

This module manages SQLAlchemy engine creation, session factory setup,
and provides utilities for safe database session management. It implements
the data access infrastructure layer with proper connection handling,
transaction management, and error handling.

The module uses SQLAlchemy 2.0 style with modern patterns for reliability
and performance in production environments.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .config import APP_CONFIG


class DatabaseManager:
    """
    Central database manager handling engine and session lifecycle.
    
    This class encapsulates all database connection logic, providing:
    - Engine creation with optimized connection pooling
    - Session factory for request-scoped database operations
    - Context manager for safe session handling with automatic cleanup
    - Connection validation utilities
    
    Attributes:
        _engine (Engine): SQLAlchemy engine instance for database connectivity.
        _session_factory (sessionmaker): Factory for creating database sessions.
    """
    
    def __init__(self) -> None:
        """
        Initialize DatabaseManager with engine and session factory.
        
        Creates the SQLAlchemy engine using configuration from APP_CONFIG.
        The engine is configured with sensible defaults for PostgreSQL connectivity.
        """
        # Create SQLAlchemy engine with connection pooling
        # pool_pre_ping ensures connections are valid before use (handles stale connections)
        self._engine: Engine = create_engine(
            APP_CONFIG.database.database_url,
            pool_pre_ping=True,      # Validate connections before checkout
            pool_size=5,             # Number of persistent connections
            max_overflow=10,         # Additional connections beyond pool_size
            echo=False,              # Set to True for SQL query logging in debug
            future=True              # Use SQLAlchemy 2.0 style
        )
        
        # Create session factory bound to the engine
        # autocommit=False: Explicit transaction control
        # autoflush=False: Manual flush control for better performance
        # expire_on_commit=False: Keep objects usable after commit
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
    
    @property
    def engine(self) -> Engine:
        """
        Get the SQLAlchemy engine instance.
        
        Returns:
            Engine: Active database engine for direct SQL operations if needed.
        """
        return self._engine
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for safe database session handling.
        
        This method provides automatic session lifecycle management:
        - Creates a new session from the factory
        - Yields session for use in database operations
        - Commits transaction on successful completion
        - Rolls back transaction on any exception
        - Always closes session regardless of outcome
        
        Yields:
            Session: SQLAlchemy session for database operations.
            
        Raises:
            SQLAlchemyError: If database operations fail.
            Exception: Re-raises any other exceptions after rollback.
            
        Example:
            with db_manager.get_session() as session:
                result = session.execute(text("SELECT * FROM healthcare_records LIMIT 1"))
        """
        # Create a new session instance
        session: Session = self._session_factory()
        
        try:
            # Yield session to caller for database operations
            yield session
            
            # Commit transaction if all operations succeed
            session.commit()
            
        except SQLAlchemyError as db_error:
            # Rollback transaction on database-specific errors
            session.rollback()
            # Re-raise database error for handling by caller
            raise db_error
            
        except Exception as general_error:
            # Rollback transaction on any other errors
            session.rollback()
            # Re-raise general error for handling by caller
            raise general_error
            
        finally:
            # Always close session to release connection back to pool
            session.close()
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Executes a simple query to verify that the database is accessible
        and credentials are valid. Useful for health checks and startup validation.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            # Attempt to execute a simple query
            with self.get_session() as session:
                # Execute trivial query to test connectivity
                result = session.execute(text("SELECT 1"))
                result.scalar()
            return True
            
        except SQLAlchemyError:
            # Connection failed due to database error
            return False
        except Exception:
            # Connection failed due to other error
            return False
    
    def dispose(self) -> None:
        """
        Dispose of the engine and close all connections.
        
        Should be called on application shutdown to properly release
        all database connections from the pool.
        """
        if self._engine is not None:
            self._engine.dispose()


# Global database manager instance
# Shared across the application for consistent connection management
db_manager = DatabaseManager()
