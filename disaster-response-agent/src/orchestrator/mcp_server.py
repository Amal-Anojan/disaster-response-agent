from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import json
import logging
import uuid
from datetime import datetime
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from ..vision_service.damage_analyzer import CerebrasVisionAnalyzer
from ..llm_service.action_generator import EmergencyActionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Disaster Response MCP Gateway",
    description="Multi-Modal AI Emergency Response System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
vision_analyzer = CerebrasVisionAnalyzer()
action_generator = EmergencyActionGenerator()

# In-memory storage for active incidents (use database in production)
active_incidents: Dict[str, Dict[str, Any]] = {}

# Request/Response Models
class EmergencyIncidentRequest(BaseModel):
    text_content: Optional[str] = None
    image_url: Optional[str] = None
    location: Optional[Dict[str, float]] = None  # {"lat": 40.7128, "lng": -74.0060}
    source: str = "manual"  # manual, social_media, sensor, 911_call
    reporter_info: Optional[Dict[str, Any]] = None
    priority_override: Optional[str] = None  # LOW, MEDIUM, HIGH, CRITICAL

class EmergencyResponse(BaseModel):
    incident_id: str
    processing_status: str  # PROCESSING, COMPLETED, ERROR
    analysis_results: Dict[str, Any]
    action_plan: Dict[str, Any]
    estimated_response_time: int  # in minutes
    severity_score: float
    confidence_score: float
    timestamp: str

class IncidentStatusUpdate(BaseModel):
    status: str
    notes: Optional[str] = None
    responder_id: Optional[str] = None

# Utility functions
def generate_incident_id() -> str:
    """Generate unique incident ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    unique_id = str(uuid.uuid4())[:8]
    return f"INC_{timestamp}_{unique_id}"

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Disaster Response MCP Gateway",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "report_incident": "/api/v1/incident/report",
            "get_status": "/api/v1/incident/status/{incident_id}",
            "active_incidents": "/api/v1/incidents/active",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "vision_analyzer": "operational" if vision_analyzer.client else "fallback_mode",
            "action_generator": "operational" if action_generator.gemini_model else "fallback_mode",
            "database": "memory_storage"  # Use "operational" when using real DB
        }
    }

@app.post("/api/v1/incident/report", response_model=EmergencyResponse)
async def report_emergency_incident(
    background_tasks: BackgroundTasks,
    incident: EmergencyIncidentRequest,
    image: UploadFile = File(None)
):
    incident_id = generate_incident_id()
    incident_data = incident.dict()
    incident_data["incident_id"] = incident_id
    incident_data["timestamp"] = datetime.now().isoformat()
    incident_data["status"] = "PROCESSING"
    """Process incoming emergency incident report"""
    try:
        incident_id = generate_incident_id()
        
        # Prepare incident data
        incident_data = {
            "incident_id": incident_id,
            "text_content": incident.text_content,
            "location": incident.location,
            "source": incident.source,
            "reporter_info": incident.reporter_info,
            "priority_override": incident.priority_override,
            "timestamp": datetime.now().isoformat(),
            "status": "PROCESSING"
        }
        
        # Handle image upload
        image_data = None
        if image:
            try:
                image_data = await image.read()
                # Save image temporarily
                temp_path = f"data/temp_images/{incident_id}_{image.filename}"
                Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(image_data)
                incident_data["image_path"] = temp_path
            except Exception as e:
                logger.warning(f"Failed to save uploaded image: {e}")
        
        # Store incident
        active_incidents[incident_id] = incident_data
        
        # Start background processing
        background_tasks.add_task(
            process_emergency_incident,
            incident_id,
            incident_data,
            image_data
        )
        
        # Return immediate response
        return EmergencyResponse(
            incident_id=incident_id,
            processing_status="PROCESSING",
            analysis_results={},
            action_plan={},
            estimated_response_time=0,
            severity_score=0.0,
            confidence_score=0.0,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error reporting incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/incident/status/{incident_id}")
async def get_incident_status(incident_id: str):
    """Get current status of emergency incident"""
    if incident_id not in active_incidents:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    incident = active_incidents[incident_id]
    return {
        "incident_id": incident_id,
        "status": incident.get("status", "UNKNOWN"),
        "timestamp": incident.get("timestamp"),
        "analysis_complete": incident.get("analysis_complete", False),
        "action_plan_generated": incident.get("action_plan_generated", False),
        "severity": incident.get("severity", 0),
        "estimated_response_time": incident.get("estimated_response_time", 0)
    }

@app.get("/api/v1/incidents/active")
async def get_active_incidents(
    limit: int = 50,
    severity_filter: Optional[int] = None,
    disaster_type: Optional[str] = None
):
    """Get list of active emergency incidents"""
    incidents = list(active_incidents.values())
    
    # Apply filters
    if severity_filter:
        incidents = [inc for inc in incidents if inc.get("severity", 0) >= severity_filter]
    
    if disaster_type:
        incidents = [inc for inc in incidents if inc.get("disaster_type", "").lower() == disaster_type.lower()]
    
    # Sort by severity and timestamp
    incidents.sort(key=lambda x: (x.get("severity", 0), x.get("timestamp", "")), reverse=True)
    
    return {
        "total_active": len(incidents),
        "incidents": incidents[:limit],
        "last_updated": datetime.now().isoformat()
    }

@app.put("/api/v1/incident/{incident_id}/status")
async def update_incident_status(incident_id: str, update: IncidentStatusUpdate):
    """Update incident status"""
    if incident_id not in active_incidents:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    incident = active_incidents[incident_id]
    incident["status"] = update.status
    incident["last_updated"] = datetime.now().isoformat()
    
    if update.notes:
        incident["status_notes"] = update.notes
    if update.responder_id:
        incident["responder_id"] = update.responder_id
    
    return {"success": True, "message": f"Incident {incident_id} status updated to {update.status}"}

@app.post("/api/v1/incidents/batch")
async def batch_process_incidents(incidents: List[EmergencyIncidentRequest]):
    """Process multiple incidents in batch"""
    if len(incidents) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 incidents")
    
    batch_results = []
    tasks = []
    
    for incident in incidents:
        incident_id = generate_incident_id()
        incident_data = {
            "incident_id": incident_id,
            "text_content": incident.text_content,
            "location": incident.location,
            "source": incident.source,
            "timestamp": datetime.now().isoformat()
        }
        
        active_incidents[incident_id] = incident_data
        
        # Create processing task
        task = asyncio.create_task(
            process_emergency_incident(incident_id, incident_data, None)
        )
        tasks.append((incident_id, task))
    
    # Wait for all tasks to complete
    for incident_id, task in tasks:
        try:
            result = await task
            batch_results.append({"incident_id": incident_id, "status": "completed"})
        except Exception as e:
            batch_results.append({"incident_id": incident_id, "status": "error", "error": str(e)})
    
    return {
        "batch_id": str(uuid.uuid4()),
        "processed": len(batch_results),
        "results": batch_results
    }

# Background processing function
async def process_emergency_incident(
    incident_id: str,
    incident_data: Dict[str, Any],
    image_data: Optional[bytes]
):
    """Process emergency incident in background"""
    try:
        logger.info(f"Processing incident {incident_id}")
        
        # Update status
        active_incidents[incident_id]["status"] = "ANALYZING"
        
        # Vision analysis (if image provided)
        vision_results = {}
        if image_data:
            try:
                vision_results = await vision_analyzer.analyze_disaster_image(
                    image_data, 
                    metadata=incident_data
                )
                logger.info(f"Vision analysis completed for {incident_id}")
            except Exception as e:
                logger.error(f"Vision analysis failed for {incident_id}: {e}")
                vision_results = {"error": str(e)}
        else:
            # Generate basic analysis from text
            vision_results = {
                "severity": 5,
                "disaster_type": "unknown",
                "urgency": "MEDIUM",
                "confidence": 0.5,
                "note": "No image provided - text-only analysis"
            }
        
        # Generate action plan
        try:
            action_plan = await action_generator.generate_action_plan(
                vision_results,
                incident_data.get("text_content", ""),
                incident_data.get("location")
            )
            logger.info(f"Action plan generated for {incident_id}")
        except Exception as e:
            logger.error(f"Action plan generation failed for {incident_id}: {e}")
            action_plan = {"error": str(e)}
        
        # Update incident with results
        active_incidents[incident_id].update({
            "status": "COMPLETED",
            "analysis_complete": True,
            "action_plan_generated": True,
            "vision_analysis": vision_results,
            "action_plan": action_plan,
            "severity": vision_results.get("severity", 5),
            "disaster_type": vision_results.get("disaster_type", "unknown"),
            "urgency": vision_results.get("urgency", "MEDIUM"),
            "estimated_response_time": action_plan.get("metrics", {}).get("estimated_response_time_minutes", 15),
            "confidence": vision_results.get("confidence", 0.5),
            "processing_completed_at": datetime.now().isoformat()
        })
        
        logger.info(f"Incident {incident_id} processing completed")
        
    except Exception as e:
        logger.error(f"Error processing incident {incident_id}: {e}")
        active_incidents[incident_id].update({
            "status": "ERROR",
            "error": str(e),
            "error_timestamp": datetime.now().isoformat()
        })

# Cleanup endpoint (for demo purposes)
@app.delete("/api/v1/incidents/cleanup")
async def cleanup_old_incidents():
    """Clean up old incidents (demo purposes)"""
    global active_incidents
    
    # In production, this would archive to database
    initial_count = len(active_incidents)
    
    # Keep only recent incidents (for demo)
    current_time = datetime.now()
    recent_incidents = {}
    
    for incident_id, incident in active_incidents.items():
        try:
            incident_time = datetime.fromisoformat(incident.get("timestamp", current_time.isoformat()))
            time_diff = (current_time - incident_time).total_seconds()
            
            # Keep incidents from last 2 hours
            if time_diff < 7200:
                recent_incidents[incident_id] = incident
        except:
            # Keep incidents with invalid timestamps
            recent_incidents[incident_id] = incident
    
    active_incidents = recent_incidents
    cleaned_count = initial_count - len(active_incidents)
    
    return {
        "message": f"Cleaned up {cleaned_count} old incidents",
        "remaining_incidents": len(active_incidents)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)