import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("Google Generative AI not available. Install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available. Install with: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

class EmergencyActionGenerator:
    def __init__(self, knowledge_base_path: str = "data/knowledge_base/"):
        self.knowledge_base_path = knowledge_base_path
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
        
        # Initialize Gemini if available
        if GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load emergency response procedures and contacts"""
        try:
            # Try to load from file first
            kb_file = os.path.join(self.knowledge_base_path, "emergency_procedures.json")
            if os.path.exists(kb_file):
                with open(kb_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load knowledge base from file: {e}")
        
        # Fallback to hardcoded knowledge base
        return {
            "procedures": {
                "flood": [
                    "Evacuate low-lying areas immediately",
                    "Deploy swift water rescue teams", 
                    "Set up emergency shelters on high ground",
                    "Coordinate with utility companies for power shutoffs",
                    "Establish communication with downstream areas"
                ],
                "fire": [
                    "Deploy fire suppression units with appropriate foam/water",
                    "Establish evacuation perimeter based on wind direction",
                    "Call additional mutual aid from neighboring departments",
                    "Set up incident command post at safe distance",
                    "Coordinate with utility companies for gas/power shutoffs"
                ],
                "earthquake": [
                    "Deploy urban search and rescue teams",
                    "Conduct rapid damage assessment of critical infrastructure", 
                    "Set up triage areas and medical stations",
                    "Inspect bridges and overpasses for structural integrity",
                    "Activate mutual aid agreements with neighboring jurisdictions"
                ],
                "storm": [
                    "Clear roads of debris and fallen trees",
                    "Restore power to critical facilities first",
                    "Set up emergency shelters for displaced residents",
                    "Deploy emergency generators to critical infrastructure",
                    "Coordinate with utility companies for power restoration"
                ]
            },
            "contacts": {
                "fire_department": "+1-555-FIRE-911",
                "police_department": "+1-555-POLICE-911",
                "medical_services": "+1-555-MEDICAL-911",
                "emergency_management": "+1-555-EMERGENCY",
                "utility_company": "+1-555-UTILITY",
                "red_cross": "+1-555-REDCROSS"
            },
            "resources": {
                "fire": ["fire_engines", "ladder_trucks", "hazmat_units", "paramedics"],
                "flood": ["rescue_boats", "high_water_vehicles", "emergency_shelters", "water_pumps"],
                "earthquake": ["search_rescue_teams", "structural_engineers", "medical_triage", "heavy_equipment"],
                "storm": ["tree_removal_crews", "power_restoration_teams", "emergency_generators", "shelter_supplies"]
            }
        }
    
    async def generate_action_plan(self, 
                                 vision_analysis: Dict[str, Any], 
                                 text_content: str = "",
                                 location: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Generate comprehensive emergency action plan"""
        try:
            # Retrieve relevant procedures
            context = self._retrieve_context(vision_analysis)
            
            # Build comprehensive prompt
            prompt = self._build_action_plan_prompt(vision_analysis, text_content, context, location)
            
            # Generate action plan using available LLM
            if self.gemini_model:
                response = await self._generate_with_gemini(prompt)
            else:
                response = await self._generate_fallback_plan(vision_analysis, context)
            
            # Parse and validate the response
            parsed_plan = self._parse_action_plan(response)
            
            return self._enhance_action_plan(parsed_plan, vision_analysis, context)
            
        except Exception as e:
            logger.error(f"Error generating action plan: {e}")
            return self._generate_fallback_plan(vision_analysis, self._retrieve_context(vision_analysis))
    
    def _build_action_plan_prompt(self, 
                                vision_analysis: Dict[str, Any],
                                text_content: str,
                                context: str,
                                location: Optional[Dict[str, float]]) -> str:
        """Build detailed prompt for action plan generation"""
        
        location_info = ""
        if location:
            location_info = f"Location: {location.get('lat', 'Unknown')}, {location.get('lng', 'Unknown')}"
        
        return f"""
EMERGENCY SITUATION ANALYSIS:
- Disaster Type: {vision_analysis.get('disaster_type', 'unknown')}
- Severity Level: {vision_analysis.get('severity', 5)}/10
- Urgency: {vision_analysis.get('urgency', 'MEDIUM')}
- Affected Infrastructure: {', '.join(vision_analysis.get('affected_infrastructure', []))}
- Estimated Affected Population: {vision_analysis.get('estimated_population', 'Unknown')}
- Confidence Level: {vision_analysis.get('confidence', 0.7):.2f}
- {location_info}

ADDITIONAL CONTEXT:
{text_content}

RELEVANT EMERGENCY PROCEDURES:
{context}

GENERATE COMPREHENSIVE EMERGENCY ACTION PLAN:

Provide a structured response plan with:

1. IMMEDIATE ACTIONS (0-30 minutes):
   - List 3-5 critical actions that must be taken immediately
   - Include evacuation procedures if needed
   - Specify safety measures

2. RESOURCE REQUIREMENTS:
   - Personnel needed (number and type)
   - Equipment required
   - Vehicles and special equipment

3. RESPONSE TIMELINE:
   - 0-30 minutes: Immediate response
   - 30-60 minutes: Secondary response  
   - 1-4 hours: Extended operations
   - 4+ hours: Recovery operations

4. COMMUNICATION PRIORITIES:
   - Who to contact first
   - Public notifications needed
   - Media coordination

5. ESTIMATED RESPONSE METRICS:
   - Total response time estimate
   - Number of personnel required
   - Cost estimate (if applicable)

Format the response as a structured JSON with the following structure:
{{
    "immediate_actions": ["action1", "action2", ...],
    "resource_requirements": {{
        "personnel": {{"type": "count"}},
        "equipment": ["item1", "item2", ...],
        "vehicles": ["vehicle1", "vehicle2", ...]
    }},
    "timeline": {{
        "0-30_min": ["action1", "action2", ...],
        "30-60_min": ["action1", "action2", ...],
        "1-4_hours": ["action1", "action2", ...],
        "4plus_hours": ["action1", "action2", ...]
    }},
    "communications": {{
        "primary_contacts": ["contact1", "contact2", ...],
        "public_notifications": ["notification1", "notification2", ...] 
    }},
    "metrics": {{
        "estimated_response_time_minutes": <number>,
        "total_personnel_required": <number>,
        "estimated_cost": <number>
    }}
}}
"""
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Google Gemini"""
        try:
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise e
    
    def _retrieve_context(self, analysis: Dict[str, Any]) -> str:
        """Retrieve relevant emergency procedures from knowledge base"""
        disaster_type = analysis.get('disaster_type', 'unknown').lower()
        severity = analysis.get('severity', 5)
        
        relevant_procedures = []
        
        # Get specific procedures for disaster type
        if disaster_type in self.knowledge_base['procedures']:
            procedures = self.knowledge_base['procedures'][disaster_type]
            relevant_procedures.extend(procedures[:4])  # Top 4 most relevant
        
        # Add general emergency procedures for high severity
        if severity >= 7:
            general_procedures = [
                "Establish incident command system",
                "Request additional mutual aid resources", 
                "Activate emergency operations center",
                "Implement emergency public information protocols"
            ]
            relevant_procedures.extend(general_procedures[:2])
        
        return "\\n".join([f"- {proc}" for proc in relevant_procedures])
    
    def _parse_action_plan(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured action plan"""
        try:
            # Try to extract JSON from response
            import re
            
            # Find JSON block in response
            json_match = re.search(r'```json\\s*(.*?)\\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = response[start:end]
                else:
                    raise ValueError("No JSON found in response")
            
            return json.loads(json_str)
            
        except Exception as e:
            logger.warning(f"Failed to parse action plan JSON: {e}")
            return self._extract_fallback_plan(response)
    
    def _extract_fallback_plan(self, response: str) -> Dict[str, Any]:
        """Extract action plan from unstructured text response"""
        lines = response.split('\\n')
        
        return {
            "immediate_actions": [
                "Assess immediate safety threats",
                "Deploy first response teams", 
                "Establish communication channels"
            ],
            "resource_requirements": {
                "personnel": {"emergency_responders": 5, "coordinators": 2},
                "equipment": ["communication_devices", "safety_equipment"],
                "vehicles": ["emergency_vehicles", "command_vehicle"]
            },
            "timeline": {
                "0-30_min": ["Initial response deployment"],
                "30-60_min": ["Situation assessment"],
                "1-4_hours": ["Extended operations"], 
                "4plus_hours": ["Recovery planning"]
            },
            "communications": {
                "primary_contacts": ["emergency_dispatch", "incident_commander"],
                "public_notifications": ["safety_alerts", "evacuation_notices"]
            },
            "metrics": {
                "estimated_response_time_minutes": 15,
                "total_personnel_required": 10,
                "estimated_cost": 50000
            }
        }
    
    def _enhance_action_plan(self, 
                           plan: Dict[str, Any], 
                           analysis: Dict[str, Any],
                           context: str) -> Dict[str, Any]:
        """Enhance and validate the generated action plan"""
        
        # Add metadata
        plan['generated_at'] = datetime.now().isoformat()
        plan['disaster_analysis'] = {
            'severity': analysis.get('severity', 5),
            'disaster_type': analysis.get('disaster_type', 'unknown'),
            'urgency': analysis.get('urgency', 'MEDIUM')
        }
        
        # Validate and adjust timeline based on severity
        severity = analysis.get('severity', 5)
        if severity >= 8:
            # Critical situation - reduce response times
            if 'metrics' in plan:
                plan['metrics']['estimated_response_time_minutes'] = max(
                    5, plan['metrics'].get('estimated_response_time_minutes', 15) - 5
                )
        
        # Add confidence score
        plan['confidence'] = analysis.get('confidence', 0.7)
        
        return plan
    
    async def _generate_fallback_plan(self, 
                                    analysis: Dict[str, Any], 
                                    context: str) -> Dict[str, Any]:
        """Generate fallback action plan when LLM is unavailable"""
        
        disaster_type = analysis.get('disaster_type', 'unknown')
        severity = analysis.get('severity', 5)
        
        # Base plan structure
        base_plan = {
            "immediate_actions": [
                f"Deploy {disaster_type} response teams",
                "Establish incident command post",
                "Assess immediate safety threats",
                "Begin evacuation if necessary"
            ],
            "resource_requirements": {
                "personnel": {
                    "incident_commander": 1,
                    "emergency_responders": max(3, severity),
                    "support_staff": 2
                },
                "equipment": self.knowledge_base['resources'].get(disaster_type, ["basic_equipment"]),
                "vehicles": ["command_vehicle", "emergency_vehicles"]
            },
            "timeline": {
                "0-30_min": ["Deploy initial response", "Establish communications"],
                "30-60_min": ["Situation assessment", "Resource deployment"],
                "1-4_hours": ["Extended operations", "Additional resources"], 
                "4plus_hours": ["Recovery operations", "Damage assessment"]
            },
            "communications": {
                "primary_contacts": ["emergency_dispatch", "fire_department", "police"],
                "public_notifications": ["emergency_alerts", "safety_instructions"]
            },
            "metrics": {
                "estimated_response_time_minutes": max(5, 20 - severity),
                "total_personnel_required": max(5, severity * 2),
                "estimated_cost": severity * 10000
            }
        }
        
        # Enhance based on disaster type
        if disaster_type == 'fire':
            base_plan['immediate_actions'].extend([
                "Deploy fire suppression units",
                "Establish evacuation perimeter"
            ])
        elif disaster_type == 'flood':
            base_plan['immediate_actions'].extend([
                "Deploy swift water rescue teams",
                "Set up emergency shelters"
            ])
        elif disaster_type == 'earthquake':
            base_plan['immediate_actions'].extend([
                "Deploy search and rescue teams",
                "Assess structural damage"
            ])
        
        return base_plan

# Test function
async def test_action_generator():
    """Test the action generator with sample data"""
    generator = EmergencyActionGenerator()
    
    # Sample vision analysis
    sample_analysis = {
        'severity': 7,
        'disaster_type': 'fire',
        'affected_infrastructure': ['residential_buildings', 'power_lines'],
        'estimated_population': 500,
        'urgency': 'HIGH',
        'confidence': 0.85
    }
    
    # Generate action plan
    plan = await generator.generate_action_plan(
        sample_analysis, 
        text_content="House fire spreading rapidly in residential area"
    )
    
    print("Generated Action Plan:")
    print(f"- Response Time: {plan['metrics']['estimated_response_time_minutes']} minutes")
    print(f"- Personnel Required: {plan['metrics']['total_personnel_required']}")
    print(f"- Immediate Actions: {len(plan['immediate_actions'])}")
    
    return plan

if __name__ == "__main__":
    asyncio.run(test_action_generator())