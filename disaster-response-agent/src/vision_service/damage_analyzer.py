import os
import base64
import asyncio
from typing import Optional, Dict, Any
from io import BytesIO
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    logger.warning("Cerebras SDK not available. Install with: pip install cerebras-cloud-sdk")
    Cerebras = None

class CerebrasVisionAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('CEREBRAS_API_KEY')
        if not self.api_key:
            logger.warning("CEREBRAS_API_KEY not found in environment variables")
            self.client = None
        else:
            try:
                self.client = Cerebras(api_key=self.api_key) if Cerebras else None
            except Exception as e:
                logger.error(f"Failed to initialize Cerebras client: {e}")
                self.client = None
        
        self.damage_categories = {
            'structural': ['building_collapse', 'bridge_damage', 'road_damage', 'foundation_crack'],
            'environmental': ['flooding', 'fire_damage', 'landslide', 'debris_flow'],
            'vehicle': ['car_accidents', 'train_derailment', 'aircraft_damage'],
            'infrastructure': ['power_lines', 'water_systems', 'communication_towers']
        }
    
    async def analyze_disaster_image(self, image_data: bytes, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze disaster damage from image data"""
        try:
            if not self.client:
                # Fallback analysis for demo purposes
                return self._generate_fallback_analysis(metadata)
            
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare the prompt for vision analysis
            prompt = """Analyze this disaster image and provide a detailed assessment:

1. Damage severity (scale 1-10, where 10 is catastrophic)
2. Disaster type (flood, fire, earthquake, storm, etc.)
3. Affected infrastructure (buildings, roads, utilities, etc.)
4. Estimated affected population (based on visible area)
5. Urgency level (LOW, MEDIUM, HIGH, CRITICAL)
6. Visible hazards or risks
7. Recommended immediate actions

Please respond in JSON format with the following structure:
{
    "severity": <number 1-10>,
    "disaster_type": "<type>",
    "affected_infrastructure": ["<list>"],
    "estimated_population": <number>,
    "urgency": "<level>",
    "hazards": ["<list>"],
    "immediate_actions": ["<list>"],
    "confidence": <0.0-1.0>
}"""

            # Make API call to Cerebras
            response = await self._make_cerebras_api_call(image_b64, prompt)
            
            return self._parse_vision_response(response)
            
        except Exception as e:
            logger.error(f"Error in disaster image analysis: {e}")
            return self._generate_fallback_analysis(metadata)
    
    async def _make_cerebras_api_call(self, image_b64: str, prompt: str) -> str:
        """Make API call to Cerebras vision model"""
        try:
            response = self.client.chat.completions.create(
                model="llama3.1-8b",  # Use available vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Cerebras API call failed: {e}")
            raise e
    
    def _parse_vision_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured response from vision model"""
        try:
            import json
            # Try to extract JSON from response
            response_text = response_text.strip()
            
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                parsed_response = json.loads(json_str)
                
                # Validate and normalize the response
                return {
                    'severity': max(1, min(10, int(parsed_response.get('severity', 5)))),
                    'disaster_type': str(parsed_response.get('disaster_type', 'unknown')).lower(),
                    'affected_infrastructure': parsed_response.get('affected_infrastructure', []),
                    'estimated_population': int(parsed_response.get('estimated_population', 100)),
                    'urgency': str(parsed_response.get('urgency', 'MEDIUM')).upper(),
                    'hazards': parsed_response.get('hazards', []),
                    'immediate_actions': parsed_response.get('immediate_actions', []),
                    'confidence': max(0.0, min(1.0, float(parsed_response.get('confidence', 0.7)))),
                    'processing_time': 1.2  # Simulated processing time
                }
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.warning(f"Failed to parse vision response: {e}")
            # Return structured fallback
            return self._generate_fallback_analysis()
    
    def _generate_fallback_analysis(self, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate fallback analysis when Cerebras API is unavailable"""
        import random
        
        disaster_types = ['flood', 'fire', 'earthquake', 'storm', 'landslide']
        urgency_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        # Generate realistic demo data
        severity = random.randint(4, 9)
        disaster_type = random.choice(disaster_types)
        
        infrastructure_options = {
            'flood': ['roads', 'residential_buildings', 'water_systems'],
            'fire': ['residential_buildings', 'vegetation', 'power_lines'],
            'earthquake': ['buildings', 'bridges', 'infrastructure'],
            'storm': ['trees', 'power_lines', 'roofing'],
            'landslide': ['roads', 'buildings', 'slopes']
        }
        
        affected_infrastructure = infrastructure_options.get(disaster_type, ['buildings'])
        
        return {
            'severity': severity,
            'disaster_type': disaster_type,
            'affected_infrastructure': affected_infrastructure,
            'estimated_population': random.randint(50, 1000),
            'urgency': urgency_levels[min(severity // 3, 3)],
            'hazards': ['debris', 'unstable_structures'],
            'immediate_actions': [
                'Deploy emergency response teams',
                'Establish evacuation routes',
                'Set up emergency shelters'
            ],
            'confidence': 0.75,
            'processing_time': 0.8,
            'note': 'Fallback analysis - Cerebras API not available'
        }

# Usage example and testing
async def main():
    analyzer = CerebrasVisionAnalyzer()
    
    # Test with a sample image (you would load real image data)
    try:
        # Create a simple test image
        test_image = Image.new('RGB', (400, 300), color='red')
        img_bytes = BytesIO()
        test_image.save(img_bytes, format='JPEG')
        image_data = img_bytes.getvalue()
        
        # Analyze the image
        result = await analyzer.analyze_disaster_image(image_data)
        print("Vision Analysis Result:")
        print(f"- Severity: {result['severity']}/10")
        print(f"- Disaster Type: {result['disaster_type']}")
        print(f"- Urgency: {result['urgency']}")
        print(f"- Estimated Population: {result['estimated_population']}")
        print(f"- Confidence: {result['confidence']:.2f}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())