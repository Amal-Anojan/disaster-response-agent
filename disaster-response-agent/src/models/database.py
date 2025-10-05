import os
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import sqlite3
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

class DatabaseManager:
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection URL. If None, uses SQLite in-memory database
        """
        if database_url is None:
            # Default to SQLite for development
            self.database_url = "sqlite:///data/disaster_response.db"
        else:
            self.database_url = database_url
        
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            # Create database directory if using SQLite
            if self.database_url.startswith("sqlite"):
                db_path = self.database_url.replace("sqlite:///", "")
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Create engine
            if self.database_url.startswith("sqlite"):
                # SQLite specific configuration
                self.engine = create_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 30
                    },
                    echo=False  # Set to True for SQL debugging
                )
            else:
                # PostgreSQL or other databases
                self.engine = create_engine(
                    self.database_url,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False
                )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create all tables
            self._create_tables()
            
            logger.info(f"Database initialized successfully: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables"""
        try:
            # Import models to register them with Base
            from .incident_model import Incident, IncidentUpdate, ResourceAllocation, ResponseTeam
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        try:
            with self.engine.connect() as connection:
                # Indexes for incidents table
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents (timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_incidents_disaster_type ON incidents (disaster_type)",
                    "CREATE INDEX IF NOT EXISTS idx_incidents_urgency ON incidents (urgency)",
                    "CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents (status)",
                    "CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents (severity)",
                    
                    # Indexes for incident_updates table
                    "CREATE INDEX IF NOT EXISTS idx_updates_incident_id ON incident_updates (incident_id)",
                    "CREATE INDEX IF NOT EXISTS idx_updates_timestamp ON incident_updates (update_timestamp)",
                    
                    # Indexes for resource_allocations table
                    "CREATE INDEX IF NOT EXISTS idx_allocations_incident_id ON resource_allocations (incident_id)",
                    "CREATE INDEX IF NOT EXISTS idx_allocations_status ON resource_allocations (allocation_status)",
                    
                    # Indexes for response_teams table
                    "CREATE INDEX IF NOT EXISTS idx_teams_status ON response_teams (status)",
                    "CREATE INDEX IF NOT EXISTS idx_teams_team_type ON response_teams (team_type)"
                ]
                
                for index_sql in indexes:
                    try:
                        connection.execute(text(index_sql))
                        connection.commit()
                    except Exception as e:
                        logger.warning(f"Failed to create index: {e}")
                
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create some database indexes: {e}")
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_engine(self):
        """Get database engine"""
        return self.engine
    
    def check_connection(self) -> bool:
        """Check if database connection is working"""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information"""
        try:
            with self.engine.connect() as connection:
                # Get database version and info
                if self.database_url.startswith("sqlite"):
                    result = connection.execute(text("SELECT sqlite_version()")).fetchone()
                    db_version = result[0] if result else "Unknown"
                    db_type = "SQLite"
                elif "postgresql" in self.database_url:
                    result = connection.execute(text("SELECT version()")).fetchone()
                    db_version = result[0] if result else "Unknown"
                    db_type = "PostgreSQL"
                else:
                    db_version = "Unknown"
                    db_type = "Unknown"
                
                return {
                    "database_type": db_type,
                    "database_version": db_version,
                    "database_url": self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url,
                    "connection_status": "Connected",
                    "tables_count": len(Base.metadata.tables)
                }
        except Exception as e:
            return {
                "database_type": "Unknown",
                "database_version": "Unknown", 
                "database_url": self.database_url,
                "connection_status": f"Error: {str(e)}",
                "tables_count": 0
            }
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute a raw SQL query"""
        try:
            with self.engine.connect() as connection:
                if params:
                    result = connection.execute(text(query), params)
                else:
                    result = connection.execute(text(query))
                
                # For SELECT queries, return all results
                if query.strip().upper().startswith("SELECT"):
                    return result.fetchall()
                else:
                    connection.commit()
                    return result.rowcount
                    
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            if self.database_url.startswith("sqlite"):
                # SQLite backup
                source_path = self.database_url.replace("sqlite:///", "")
                
                # Ensure backup directory exists
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Copy database file
                import shutil
                shutil.copy2(source_path, backup_path)
                
                logger.info(f"Database backup created: {backup_path}")
                return True
            else:
                # For other databases, would need pg_dump or similar
                logger.warning("Backup not implemented for this database type")
                return False
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            if self.database_url.startswith("sqlite"):
                # SQLite restore
                target_path = self.database_url.replace("sqlite:///", "")
                
                # Close existing connections
                if self.engine:
                    self.engine.dispose()
                
                # Restore database file
                import shutil
                shutil.copy2(backup_path, target_path)
                
                # Reinitialize database
                self._initialize_database()
                
                logger.info(f"Database restored from: {backup_path}")
                return True
            else:
                logger.warning("Restore not implemented for this database type")
                return False
                
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    def get_table_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tables"""
        stats = {}
        
        try:
            for table_name, table in Base.metadata.tables.items():
                try:
                    with self.engine.connect() as connection:
                        # Get row count
                        count_result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
                        row_count = count_result[0] if count_result else 0
                        
                        stats[table_name] = {
                            "row_count": row_count,
                            "columns": list(table.columns.keys()),
                            "column_count": len(table.columns),
                            "primary_key": [col.name for col in table.primary_key.columns],
                            "has_data": row_count > 0
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to get stats for table {table_name}: {e}")
                    stats[table_name] = {
                        "row_count": -1,
                        "columns": [],
                        "column_count": 0,
                        "primary_key": [],
                        "has_data": False,
                        "error": str(e)
                    }
            
        except Exception as e:
            logger.error(f"Failed to get table statistics: {e}")
        
        return stats
    
    def cleanup_old_records(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old records from the database"""
        cleanup_stats = {}
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with self.get_session() as session:
                # Import models
                from .incident_model import Incident, IncidentUpdate
                
                # Clean up old resolved incidents
                old_incidents = session.query(Incident).filter(
                    Incident.timestamp < cutoff_date,
                    Incident.status == 'RESOLVED'
                )
                incident_count = old_incidents.count()
                old_incidents.delete(synchronize_session=False)
                cleanup_stats['incidents'] = incident_count
                
                # Clean up old incident updates
                old_updates = session.query(IncidentUpdate).filter(
                    IncidentUpdate.update_timestamp < cutoff_date
                )
                update_count = old_updates.count()
                old_updates.delete(synchronize_session=False)
                cleanup_stats['incident_updates'] = update_count
                
                session.commit()
                
            logger.info(f"Cleaned up old records: {cleanup_stats}")
            
        except Exception as e:
            logger.error(f"Cleanup operation failed: {e}")
            cleanup_stats['error'] = str(e)
        
        return cleanup_stats
    
    def close(self):
        """Close database connections"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

# Global database instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        # Get database URL from environment or use default
        database_url = os.getenv('DATABASE_URL')
        _db_manager = DatabaseManager(database_url)
    
    return _db_manager

def init_database(database_url: Optional[str] = None) -> DatabaseManager:
    """Initialize database with custom URL"""
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    return _db_manager

# Utility functions for common database operations
async def health_check() -> Dict[str, Any]:
    """Check database health"""
    try:
        db_manager = get_database_manager()
        
        # Check connection
        is_connected = db_manager.check_connection()
        
        # Get connection info
        connection_info = db_manager.get_connection_info()
        
        # Get table stats
        table_stats = db_manager.get_table_stats()
        
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "connected": is_connected,
            "connection_info": connection_info,
            "table_stats": table_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Usage example and testing
def test_database():
    """Test database functionality"""
    print("Testing Database Manager...")
    
    # Initialize database
    db_manager = DatabaseManager()
    
    # Test connection
    print(f"Connection test: {db_manager.check_connection()}")
    
    # Get connection info
    info = db_manager.get_connection_info()
    print(f"Connection info: {info}")
    
    # Get table stats
    stats = db_manager.get_table_stats()
    print(f"Table stats: {stats}")
    
    # Test health check
    import asyncio
    health = asyncio.run(health_check())
    print(f"Health check: {health}")
    
    print("Database test completed!")

if __name__ == "__main__":
    test_database()