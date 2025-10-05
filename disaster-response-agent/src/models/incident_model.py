from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

Base = declarative_base()

class Incident(Base):
    """Main incident model"""
    __tablename__ = 'incidents'
    
    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(50), unique=True, index=True, nullable=False)
    
    # Core incident data
    disaster_type = Column(String(50), nullable=False, index=True)
    severity = Column(Float, nullable=False, default=5.0)
    urgency = Column(String(20), nullable=False, default='MEDIUM', index=True)
    status = Column(String(20), nullable=False, default='PROCESSING', index=True)
    
    # Content and description
    text_content = Column(Text)
    description = Column(Text)
    
    # Location data
    latitude = Column(Float)
    longitude = Column(Float)
    location_description = Column(String(200))
    
    # Impact assessment
    estimated_affected_population = Column(Integer, default=0)
    affected_infrastructure = Column(JSON)  # List of affected infrastructure
    
    # Analysis results
    vision_analysis = Column(JSON)
    text_analysis = Column(JSON)
    combined_analysis = Column(JSON)
    action_plan = Column(JSON)
    
    # Processing metadata
    processing_status = Column(String(20), default='PENDING')
    confidence_score = Column(Float, default=0.0)
    processing_time_seconds = Column(Float)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime)
    
    # Source information
    source = Column(String(50), default='manual')
    source_metadata = Column(JSON)
    
    # Response metrics
    estimated_response_time = Column(Integer)  # in minutes
    actual_response_time = Column(Integer)     # in minutes
    response_team_count = Column(Integer, default=0)
    
    # Priority and escalation
    priority_score = Column(Float)
    escalation_level = Column(String(20), default='local')
    
    # Relationships
    updates = relationship("IncidentUpdate", back_populates="incident", cascade="all, delete-orphan")
    resource_allocations = relationship("ResourceAllocation", back_populates="incident", cascade="all, delete-orphan")
    
    def __init__(self, **kwargs):
        """Initialize incident with auto-generated ID if not provided"""
        if 'incident_id' not in kwargs:
            timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
            kwargs['incident_id'] = f"INC_{timestamp_str}_{uuid.uuid4().hex[:6]}"
        
        super().__init__(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary"""
        return {
            'id': self.id,
            'incident_id': self.incident_id,
            'disaster_type': self.disaster_type,
            'severity': self.severity,
            'urgency': self.urgency,
            'status': self.status,
            'text_content': self.text_content,
            'description': self.description,
            'location': {
                'lat': self.latitude,
                'lng': self.longitude,
                'description': self.location_description
            } if self.latitude and self.longitude else None,
            'estimated_affected_population': self.estimated_affected_population,
            'affected_infrastructure': self.affected_infrastructure or [],
            'vision_analysis': self.vision_analysis or {},
            'text_analysis': self.text_analysis or {},
            'combined_analysis': self.combined_analysis or {},
            'action_plan': self.action_plan or {},
            'processing_status': self.processing_status,
            'confidence_score': self.confidence_score,
            'processing_time_seconds': self.processing_time_seconds,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'source': self.source,
            'source_metadata': self.source_metadata or {},
            'estimated_response_time': self.estimated_response_time,
            'actual_response_time': self.actual_response_time,
            'response_team_count': self.response_team_count,
            'priority_score': self.priority_score,
            'escalation_level': self.escalation_level
        }
    
    def update_status(self, new_status: str, update_reason: str = None):
        """Update incident status with automatic timestamp"""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.utcnow()
        
        if new_status == 'RESOLVED':
            self.resolved_at = datetime.utcnow()
        
        # Create update record
        update = IncidentUpdate(
            incident_id=self.id,
            update_type='status_change',
            old_value=old_status,
            new_value=new_status,
            update_reason=update_reason or f"Status changed from {old_status} to {new_status}"
        )
        self.updates.append(update)
    
    def update_analysis(self, analysis_type: str, analysis_data: Dict[str, Any]):
        """Update analysis data"""
        if analysis_type == 'vision':
            self.vision_analysis = analysis_data
        elif analysis_type == 'text':
            self.text_analysis = analysis_data
        elif analysis_type == 'combined':
            self.combined_analysis = analysis_data
        elif analysis_type == 'action_plan':
            self.action_plan = analysis_data
        
        self.updated_at = datetime.utcnow()
        
        # Create update record
        update = IncidentUpdate(
            incident_id=self.id,
            update_type='analysis_update',
            update_data={'analysis_type': analysis_type, 'data_keys': list(analysis_data.keys())},
            update_reason=f"Updated {analysis_type} analysis"
        )
        self.updates.append(update)
    
    def calculate_priority_score(self) -> float:
        """Calculate priority score based on incident attributes"""
        # Base score from severity and urgency
        urgency_weights = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        urgency_weight = urgency_weights.get(self.urgency, 2)
        
        base_score = self.severity * urgency_weight
        
        # Population impact multiplier
        if self.estimated_affected_population > 500:
            population_multiplier = 1.5
        elif self.estimated_affected_population > 100:
            population_multiplier = 1.2
        else:
            population_multiplier = 1.0
        
        # Infrastructure impact
        infrastructure_count = len(self.affected_infrastructure or [])
        infrastructure_multiplier = 1.0 + (infrastructure_count * 0.1)
        
        # Time sensitivity (recent incidents get higher priority)
        if self.timestamp:
            hours_since = (datetime.utcnow() - self.timestamp).total_seconds() / 3600
            if hours_since < 1:
                time_multiplier = 1.3
            elif hours_since < 6:
                time_multiplier = 1.1
            else:
                time_multiplier = 1.0
        else:
            time_multiplier = 1.0
        
        # Calculate final priority score
        priority_score = base_score * population_multiplier * infrastructure_multiplier * time_multiplier
        
        # Update the stored priority score
        self.priority_score = round(priority_score, 2)
        
        return self.priority_score

class IncidentUpdate(Base):
    """Incident update/audit trail model"""
    __tablename__ = 'incident_updates'
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey('incidents.id'), nullable=False, index=True)
    
    # Update details
    update_type = Column(String(50), nullable=False)  # status_change, analysis_update, resource_update, etc.
    update_reason = Column(Text)
    
    # Change tracking
    old_value = Column(Text)
    new_value = Column(Text)
    update_data = Column(JSON)  # Additional structured data
    
    # Metadata
    update_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_by = Column(String(100))  # User or system that made the update
    update_source = Column(String(50), default='system')
    
    # Relationship
    incident = relationship("Incident", back_populates="updates")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert update to dictionary"""
        return {
            'id': self.id,
            'incident_id': self.incident_id,
            'update_type': self.update_type,
            'update_reason': self.update_reason,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'update_data': self.update_data or {},
            'update_timestamp': self.update_timestamp.isoformat() if self.update_timestamp else None,
            'updated_by': self.updated_by,
            'update_source': self.update_source
        }

class ResourceAllocation(Base):
    """Resource allocation model"""
    __tablename__ = 'resource_allocations'
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey('incidents.id'), nullable=False, index=True)
    allocation_id = Column(String(50), unique=True, nullable=False)
    
    # Allocation details
    allocated_resources = Column(JSON, nullable=False)  # List of allocated resources
    backup_resources = Column(JSON)  # Backup resources
    resource_requirements = Column(JSON)  # Original requirements
    
    # Metrics
    total_personnel = Column(Integer, default=0)
    estimated_response_time_minutes = Column(Integer)
    estimated_cost = Column(Float)
    coverage_percentage = Column(Float)
    
    # Status and timestamps
    allocation_status = Column(String(20), default='PENDING', index=True)
    allocated_at = Column(DateTime, default=datetime.utcnow)
    activated_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Performance tracking
    actual_response_time_minutes = Column(Integer)
    actual_cost = Column(Float)
    effectiveness_rating = Column(Float)  # 1-10 rating
    
    # Relationship
    incident = relationship("Incident", back_populates="resource_allocations")
    
    def __init__(self, **kwargs):
        """Initialize allocation with auto-generated ID if not provided"""
        if 'allocation_id' not in kwargs:
            timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
            kwargs['allocation_id'] = f"ALLOC_{timestamp_str}_{uuid.uuid4().hex[:6]}"
        
        super().__init__(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert allocation to dictionary"""
        return {
            'id': self.id,
            'incident_id': self.incident_id,
            'allocation_id': self.allocation_id,
            'allocated_resources': self.allocated_resources or [],
            'backup_resources': self.backup_resources or [],
            'resource_requirements': self.resource_requirements or {},
            'total_personnel': self.total_personnel,
            'estimated_response_time_minutes': self.estimated_response_time_minutes,
            'estimated_cost': self.estimated_cost,
            'coverage_percentage': self.coverage_percentage,
            'allocation_status': self.allocation_status,
            'allocated_at': self.allocated_at.isoformat() if self.allocated_at else None,
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'actual_response_time_minutes': self.actual_response_time_minutes,
            'actual_cost': self.actual_cost,
            'effectiveness_rating': self.effectiveness_rating
        }
    
    def activate(self):
        """Activate the resource allocation"""
        self.allocation_status = 'ACTIVE'
        self.activated_at = datetime.utcnow()
    
    def complete(self, effectiveness_rating: Optional[float] = None):
        """Mark allocation as completed"""
        self.allocation_status = 'COMPLETED'
        self.completed_at = datetime.utcnow()
        if effectiveness_rating is not None:
            self.effectiveness_rating = effectiveness_rating

class ResponseTeam(Base):
    """Response team model"""
    __tablename__ = 'response_teams'
    
    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(String(50), unique=True, nullable=False)
    
    # Team details
    team_name = Column(String(100), nullable=False)
    team_type = Column(String(50), nullable=False, index=True)  # fire, medical, police, etc.
    specializations = Column(JSON)  # List of specializations
    
    # Capacity and location
    max_capacity = Column(Integer, nullable=False)
    current_capacity = Column(Integer, nullable=False)
    home_location_lat = Column(Float)
    home_location_lng = Column(Float)
    current_location_lat = Column(Float)
    current_location_lng = Column(Float)
    
    # Status and availability
    status = Column(String(20), default='AVAILABLE', index=True)  # AVAILABLE, DEPLOYED, MAINTENANCE
    availability_hours = Column(JSON)  # Operating hours
    
    # Contact and communication
    primary_contact = Column(String(20))
    radio_frequency = Column(String(20))
    communication_channels = Column(JSON)
    
    # Performance metrics
    average_response_time_minutes = Column(Float)
    total_incidents_handled = Column(Integer, default=0)
    success_rate = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_deployed_at = Column(DateTime)
    
    def __init__(self, **kwargs):
        """Initialize team with auto-generated ID if not provided"""
        if 'team_id' not in kwargs:
            team_type = kwargs.get('team_type', 'GEN')
            kwargs['team_id'] = f"{team_type[:3].upper()}_{uuid.uuid4().hex[:8]}"
        
        super().__init__(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert team to dictionary"""
        return {
            'id': self.id,
            'team_id': self.team_id,
            'team_name': self.team_name,
            'team_type': self.team_type,
            'specializations': self.specializations or [],
            'max_capacity': self.max_capacity,
            'current_capacity': self.current_capacity,
            'home_location': {
                'lat': self.home_location_lat,
                'lng': self.home_location_lng
            } if self.home_location_lat and self.home_location_lng else None,
            'current_location': {
                'lat': self.current_location_lat,
                'lng': self.current_location_lng
            } if self.current_location_lat and self.current_location_lng else None,
            'status': self.status,
            'availability_hours': self.availability_hours or {},
            'primary_contact': self.primary_contact,
            'radio_frequency': self.radio_frequency,
            'communication_channels': self.communication_channels or [],
            'average_response_time_minutes': self.average_response_time_minutes,
            'total_incidents_handled': self.total_incidents_handled,
            'success_rate': self.success_rate,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_deployed_at': self.last_deployed_at.isoformat() if self.last_deployed_at else None
        }
    
    def deploy(self, incident_id: Optional[int] = None):
        """Deploy team to incident"""
        self.status = 'DEPLOYED'
        self.last_deployed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def return_to_base(self):
        """Return team to available status"""
        self.status = 'AVAILABLE'
        if self.home_location_lat and self.home_location_lng:
            self.current_location_lat = self.home_location_lat
            self.current_location_lng = self.home_location_lng
        self.updated_at = datetime.utcnow()
    
    def update_performance_metrics(self, response_time_minutes: float, success: bool):
        """Update performance metrics"""
        # Update response time (running average)
        if self.average_response_time_minutes is None:
            self.average_response_time_minutes = response_time_minutes
        else:
            # Weighted average (give more weight to recent performances)
            self.average_response_time_minutes = (
                self.average_response_time_minutes * 0.8 + response_time_minutes * 0.2
            )
        
        # Update incident count
        self.total_incidents_handled += 1
        
        # Update success rate
        if self.success_rate is None:
            self.success_rate = 1.0 if success else 0.0
        else:
            # Calculate new success rate
            total_successes = self.success_rate * (self.total_incidents_handled - 1)
            if success:
                total_successes += 1
            self.success_rate = total_successes / self.total_incidents_handled
        
        self.updated_at = datetime.utcnow()

# Database utility functions
def create_sample_data(session):
    """Create sample data for testing"""
    
    # Create sample incidents
    sample_incidents = [
        Incident(
            disaster_type='fire',
            severity=8.5,
            urgency='CRITICAL',
            text_content='Large apartment building fire with people trapped on upper floors',
            latitude=37.7749,
            longitude=-122.4194,
            estimated_affected_population=150,
            affected_infrastructure=['residential_building', 'power_lines'],
            source='emergency_call'
        ),
        Incident(
            disaster_type='flood',
            severity=6.0,
            urgency='HIGH',
            text_content='Flash flooding due to broken water main affecting downtown area',
            latitude=37.7849,
            longitude=-122.4094,
            estimated_affected_population=300,
            affected_infrastructure=['roads', 'commercial_buildings'],
            source='social_media'
        ),
        Incident(
            disaster_type='earthquake',
            severity=7.2,
            urgency='HIGH',
            text_content='Structural damage to bridge following 6.5 magnitude earthquake',
            latitude=37.7649,
            longitude=-122.4294,
            estimated_affected_population=1000,
            affected_infrastructure=['bridge', 'highway'],
            source='sensor_network'
        )
    ]
    
    # Create sample response teams
    sample_teams = [
        ResponseTeam(
            team_name='Fire Station 1 Alpha',
            team_type='fire',
            specializations=['fire_suppression', 'rescue', 'hazmat'],
            max_capacity=8,
            current_capacity=8,
            home_location_lat=37.7749,
            home_location_lng=-122.4194,
            primary_contact='+1-555-FIRE-001',
            radio_frequency='154.280'
        ),
        ResponseTeam(
            team_name='EMS Unit 2',
            team_type='medical',
            specializations=['emergency_medicine', 'trauma', 'transport'],
            max_capacity=6,
            current_capacity=6,
            home_location_lat=37.7849,
            home_location_lng=-122.4094,
            primary_contact='+1-555-EMS-002',
            radio_frequency='463.000'
        ),
        ResponseTeam(
            team_name='Police District 3',
            team_type='police',
            specializations=['traffic_control', 'security', 'crowd_management'],
            max_capacity=4,
            current_capacity=4,
            home_location_lat=37.7649,
            home_location_lng=-122.4294,
            primary_contact='+1-555-PD-003',
            radio_frequency='460.250'
        )
    ]
    
    # Add to session
    session.add_all(sample_incidents)
    session.add_all(sample_teams)
    session.commit()
    
    print(f"Created {len(sample_incidents)} sample incidents and {len(sample_teams)} sample teams")

# Usage example and testing
def test_models():
    """Test the database models"""
    from .database import get_database_manager
    
    print("Testing Database Models...")
    
    # Get database manager
    db_manager = get_database_manager()
    
    # Test creating an incident
    with db_manager.get_session() as session:
        # Create test incident
        incident = Incident(
            disaster_type='fire',
            severity=7.5,
            urgency='HIGH',
            text_content='Test fire incident',
            latitude=37.7749,
            longitude=-122.4194,
            estimated_affected_population=50
        )
        
        # Add and commit
        session.add(incident)
        session.commit()
        session.refresh(incident)
        
        print(f"Created incident: {incident.incident_id}")
        print(f"Priority score: {incident.calculate_priority_score()}")
        
        # Test updating status
        incident.update_status('DISPATCHED', 'Resources deployed')
        session.commit()
        
        print(f"Updated status to: {incident.status}")
        print(f"Number of updates: {len(incident.updates)}")
        
        # Test dictionary conversion
        incident_dict = incident.to_dict()
        print(f"Incident dict keys: {list(incident_dict.keys())}")
    
    print("Model test completed!")

if __name__ == "__main__":
    test_models()