import os
import json
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SampleDataLoader:
    """Load sample data for testing and demonstration"""
    
    def __init__(self):
        """Initialize sample data loader"""
        self.base_path = Path(__file__).parent.parent
        self.data_path = self.base_path / 'data'
        self.datasets_path = self.data_path / 'datasets'
        self.sample_images_path = self.datasets_path / 'sample_images'
        
        # Ensure directories exist
        self.datasets_path.mkdir(parents=True, exist_ok=True)
        self.sample_images_path.mkdir(parents=True, exist_ok=True)
    
    def generate_sample_incidents(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate sample incident data"""
        logger.info(f"Generating {count} sample incidents...")
        
        disaster_types = ['fire', 'flood', 'earthquake', 'accident', 'medical', 'hazmat', 'storm']
        urgency_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        statuses = ['PROCESSING', 'DISPATCHED', 'ON_SCENE', 'RESOLVED']
        
        # Base locations (San Francisco Bay Area)
        base_locations = [
            {'lat': 37.7749, 'lng': -122.4194, 'name': 'San Francisco'},
            {'lat': 37.6879, 'lng': -122.4702, 'name': 'Daly City'},
            {'lat': 37.4419, 'lng': -122.1430, 'name': 'Palo Alto'},
            {'lat': 37.3541, 'lng': -121.9552, 'name': 'San Jose'},
            {'lat': 37.8044, 'lng': -122.2712, 'name': 'Berkeley'},
            {'lat': 37.5407, 'lng': -122.2999, 'name': 'San Mateo'},
        ]
        
        infrastructure_types = [
            'residential_building', 'commercial_building', 'school', 'hospital',
            'bridge', 'highway', 'power_lines', 'water_system', 'gas_lines',
            'communications', 'public_transport', 'emergency_services'
        ]
        
        incidents = []
        
        for i in range(count):
            # Select random base location and add some variance
            base_loc = random.choice(base_locations)
            lat_offset = random.uniform(-0.02, 0.02)
            lng_offset = random.uniform(-0.02, 0.02)
            
            disaster_type = random.choice(disaster_types)
            urgency = random.choice(urgency_levels)
            severity = random.uniform(2.0, 10.0)
            
            # Generate realistic timestamp (within last 30 days)
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            
            timestamp = datetime.now()
            timestamp = timestamp.replace(
                day=timestamp.day - days_ago % 28,
                hour=hours_ago,
                minute=minutes_ago
            )
            
            # Generate description based on disaster type
            descriptions = {
                'fire': [
                    "Building fire with visible flames and heavy smoke",
                    "Wildfire spreading rapidly through residential area", 
                    "Vehicle fire blocking major intersection",
                    "Industrial fire with possible chemical involvement",
                    "Apartment complex fire with people trapped"
                ],
                'flood': [
                    "Flash flooding due to broken water main",
                    "River overflow affecting downtown area",
                    "Storm drain backup causing street flooding",
                    "Coastal flooding from high tides",
                    "Building basement flooding affecting utilities"
                ],
                'earthquake': [
                    "Structural damage from magnitude 5.2 earthquake",
                    "Bridge damage following seismic activity", 
                    "Building collapse risk after earthquake",
                    "Gas leak caused by earthquake damage",
                    "Highway overpass showing structural stress"
                ],
                'accident': [
                    "Multi-vehicle collision on Highway 101",
                    "Pedestrian accident at busy intersection",
                    "Construction accident with worker injuries",
                    "School bus involved in traffic accident",
                    "Fatal motorcycle accident on city streets"
                ],
                'medical': [
                    "Mass casualty incident at public event",
                    "Food poisoning outbreak at restaurant",
                    "Chemical exposure at workplace",
                    "Multiple injuries from building collapse",
                    "Infectious disease outbreak at school"
                ],
                'hazmat': [
                    "Chemical spill from overturned tanker truck",
                    "Gas leak at industrial facility",
                    "Hazardous material release at port",
                    "Unknown chemical odor in office building",
                    "Radioactive material transport incident"
                ],
                'storm': [
                    "Severe thunderstorm with damaging winds",
                    "Tornado warning in residential area",
                    "Hailstorm damaging vehicles and buildings",
                    "Lightning strike causing power outage",
                    "High winds bringing down power lines"
                ]
            }
            
            description = random.choice(descriptions.get(disaster_type, ["Emergency situation requiring response"]))
            
            # Generate affected infrastructure
            num_infrastructure = random.randint(1, 4)
            affected_infrastructure = random.sample(infrastructure_types, num_infrastructure)
            
            # Generate affected population based on disaster type and location
            if disaster_type in ['earthquake', 'flood', 'storm']:
                affected_population = random.randint(100, 5000)
            elif disaster_type in ['fire', 'hazmat']:
                affected_population = random.randint(20, 500)
            else:
                affected_population = random.randint(5, 100)
            
            incident = {
                'incident_id': f"INC_{timestamp.strftime('%Y%m%d')}_{i+1:04d}",
                'disaster_type': disaster_type,
                'severity': round(severity, 1),
                'urgency': urgency,
                'status': random.choice(statuses),
                'text_content': description,
                'location': {
                    'lat': round(base_loc['lat'] + lat_offset, 6),
                    'lng': round(base_loc['lng'] + lng_offset, 6),
                    'description': f"Near {base_loc['name']}"
                },
                'estimated_affected_population': affected_population,
                'affected_infrastructure': affected_infrastructure,
                'estimated_response_time': random.randint(5, 45),
                'timestamp': timestamp.isoformat(),
                'source': random.choice(['emergency_call', 'social_media', 'sensor_network', 'manual_report']),
                'confidence': round(random.uniform(0.6, 0.95), 2),
                'tags': [disaster_type, urgency.lower(), base_loc['name'].lower().replace(' ', '_')],
                'metadata': {
                    'generated': True,
                    'generated_at': datetime.now().isoformat(),
                    'scenario': f"sample_{disaster_type}_{i+1}"
                }
            }
            
            incidents.append(incident)
        
        return incidents
    
    def generate_sample_social_media_posts(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate sample social media posts"""
        logger.info(f"Generating {count} sample social media posts...")
        
        platforms = ['twitter', 'facebook', 'instagram', 'tiktok']
        
        # Sample emergency-related posts
        post_templates = [
            "🔥 FIRE on {location}! Heavy smoke visible. Stay away from the area. #Emergency #Fire",
            "Major accident on {location}. Traffic completely blocked. Multiple ambulances on scene. #Accident #Traffic",
            "EARTHQUAKE just hit! Magnitude feels like 5+. Buildings shaking in {location}. #Earthquake #Safety",
            "Flooding on {location} after water main break. Cars stuck in water. #Flood #Emergency",
            "⚠️ Chemical smell near {location}. People evacuating the area. #HazMat #Evacuation",
            "Power outage affecting entire {location} area. Traffic lights down. #PowerOutage #Safety",
            "🚨 Breaking: Building collapse at {location}. Emergency crews responding. #Emergency #Collapse",
            "Storm damage in {location}. Trees down, power lines damaged. #Storm #Damage",
            "Medical emergency at {location}. Multiple ambulances. Area blocked off. #Medical #Emergency",
            "Gas leak reported near {location}. Residents being evacuated. #GasLeak #Evacuation"
        ]
        
        locations = [
            "Main St & 1st Ave", "Downtown Plaza", "Highway 101", "Central Park",
            "University Campus", "Shopping Center", "Industrial District", "Residential Area",
            "City Hall", "Memorial Bridge", "Waterfront", "Business District"
        ]
        
        posts = []
        
        for i in range(count):
            template = random.choice(post_templates)
            location = random.choice(locations)
            
            # Generate timestamp within last 7 days
            days_ago = random.randint(0, 7)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            
            timestamp = datetime.now()
            timestamp = timestamp.replace(
                day=timestamp.day - days_ago % 28,
                hour=hours_ago,
                minute=minutes_ago
            )
            
            post = {
                'post_id': f"POST_{timestamp.strftime('%Y%m%d%H%M%S')}_{i:04d}",
                'platform': random.choice(platforms),
                'content': template.format(location=location),
                'timestamp': timestamp.isoformat(),
                'location': location,
                'author': f"user_{random.randint(1000, 9999)}",
                'engagement': {
                    'likes': random.randint(0, 1000),
                    'shares': random.randint(0, 100),
                    'comments': random.randint(0, 50)
                },
                'sentiment': random.choice(['negative', 'neutral', 'concerned', 'urgent']),
                'emergency_relevance': random.uniform(0.7, 0.95),
                'verified': random.choice([True, False]),
                'media_attached': random.choice([True, False]),
                'tags': ['emergency', 'breaking', 'local'],
                'metadata': {
                    'generated': True,
                    'generated_at': datetime.now().isoformat(),
                    'language': 'en'
                }
            }
            
            posts.append(post)
        
        return posts
    
    def generate_sample_resources(self) -> Dict[str, Any]:
        """Generate sample emergency resource data"""
        logger.info("Generating sample emergency resources...")
        
        # Fire engines
        fire_engines = []
        for i in range(8):
            engine = {
                'resource_id': f"FE{i+1:03d}",
                'type': random.choice(['Type 1 Fire Engine', 'Type 2 Fire Engine', 'Aerial Ladder']),
                'status': random.choice(['available', 'deployed', 'maintenance']),
                'crew_size': random.randint(3, 6),
                'location': {
                    'lat': round(37.7749 + random.uniform(-0.1, 0.1), 6),
                    'lng': round(-122.4194 + random.uniform(-0.1, 0.1), 6),
                    'station': f"Fire Station {random.randint(1, 20)}"
                },
                'capabilities': random.sample([
                    'fire_suppression', 'rescue', 'hazmat', 'medical',
                    'aerial_operations', 'water_rescue', 'technical_rescue'
                ], random.randint(2, 5)),
                'equipment': {
                    'water_capacity': random.randint(300, 1000),
                    'pump_capacity': random.randint(1000, 2000),
                    'ladder_reach': random.randint(0, 100) if 'Aerial' in engine.get('type', '') else 0
                }
            }
            fire_engines.append(engine)
        
        # Ambulances
        ambulances = []
        for i in range(12):
            ambulance = {
                'resource_id': f"AMB{i+1:03d}",
                'type': random.choice(['BLS Ambulance', 'ALS Ambulance', 'Critical Care Transport']),
                'status': random.choice(['available', 'deployed', 'hospital', 'maintenance']),
                'crew_size': random.randint(2, 4),
                'location': {
                    'lat': round(37.7749 + random.uniform(-0.1, 0.1), 6),
                    'lng': round(-122.4194 + random.uniform(-0.1, 0.1), 6),
                    'station': f"EMS Station {random.randint(1, 15)}"
                },
                'capabilities': random.sample([
                    'basic_life_support', 'advanced_life_support', 'critical_care',
                    'pediatric_care', 'trauma_response', 'cardiac_care'
                ], random.randint(2, 4)),
                'equipment': {
                    'cardiac_monitor': True,
                    'ventilator': random.choice([True, False]),
                    'medications': random.choice(['basic', 'advanced', 'critical'])
                }
            }
            ambulances.append(ambulance)
        
        # Police units
        police_units = []
        for i in range(15):
            unit = {
                'resource_id': f"PU{i+1:03d}",
                'type': random.choice(['Patrol Car', 'Motorcycle', 'SUV', 'K9 Unit']),
                'status': random.choice(['available', 'deployed', 'patrol', 'maintenance']),
                'crew_size': random.randint(1, 2),
                'location': {
                    'lat': round(37.7749 + random.uniform(-0.1, 0.1), 6),
                    'lng': round(-122.4194 + random.uniform(-0.1, 0.1), 6),
                    'precinct': f"Precinct {random.randint(1, 12)}"
                },
                'capabilities': random.sample([
                    'traffic_control', 'crowd_control', 'investigation',
                    'k9_operations', 'swat_support', 'emergency_response'
                ], random.randint(2, 4))
            }
            police_units.append(unit)
        
        return {
            'fire_engines': fire_engines,
            'ambulances': ambulances,
            'police_units': police_units,
            'last_updated': datetime.now().isoformat(),
            'total_resources': len(fire_engines) + len(ambulances) + len(police_units)
        }
    
    def create_sample_images(self):
        """Create sample disaster images for testing"""
        logger.info("Creating sample disaster images...")
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
        except ImportError:
            logger.warning("PIL not available. Skipping sample image creation.")
            return
        
        # Image scenarios
        scenarios = [
            {'name': 'building_fire', 'color': (255, 100, 0), 'text': 'BUILDING FIRE'},
            {'name': 'flood_damage', 'color': (0, 100, 255), 'text': 'FLOOD DAMAGE'},
            {'name': 'earthquake_damage', 'color': (139, 69, 19), 'text': 'EARTHQUAKE DAMAGE'},
            {'name': 'car_accident', 'color': (128, 128, 128), 'text': 'CAR ACCIDENT'},
            {'name': 'storm_damage', 'color': (64, 64, 64), 'text': 'STORM DAMAGE'},
        ]
        
        for scenario in scenarios:
            # Create image
            img = Image.new('RGB', (640, 480), color=scenario['color'])
            draw = ImageDraw.Draw(img)
            
            # Add some visual elements
            # Add rectangles to simulate buildings/objects
            for _ in range(random.randint(2, 5)):
                x1 = random.randint(0, 500)
                y1 = random.randint(0, 350)
                x2 = x1 + random.randint(50, 140)
                y2 = y1 + random.randint(50, 130)
                color = tuple(random.randint(0, 255) for _ in range(3))
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0))
            
            # Add text label
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((0, 0), scenario['text'], font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (640 - text_width) // 2
            y = 450 - text_height
            
            # Add text background
            draw.rectangle([x-10, y-5, x+text_width+10, y+text_height+5], 
                          fill=(255, 255, 255), outline=(0, 0, 0))
            draw.text((x, y), scenario['text'], fill=(0, 0, 0), font=font)
            
            # Save image
            img_path = self.sample_images_path / f"{scenario['name']}.jpg"
            img.save(img_path, 'JPEG', quality=85)
            logger.info(f"Created sample image: {img_path}")
    
    def save_sample_data(self):
        """Save all sample data to files"""
        logger.info("Saving sample data to files...")
        
        # Generate and save sample incidents
        incidents = self.generate_sample_incidents(50)
        incidents_file = self.datasets_path / 'sample_incidents.json'
        with open(incidents_file, 'w') as f:
            json.dump(incidents, f, indent=2, default=str)
        logger.info(f"Saved {len(incidents)} incidents to {incidents_file}")
        
        # Generate and save social media posts (update existing file)
        posts = self.generate_sample_social_media_posts(100)
        posts_file = self.datasets_path / 'social_media_posts.json'
        
        # Load existing posts if they exist
        existing_posts = []
        if posts_file.exists():
            try:
                with open(posts_file, 'r') as f:
                    existing_data = json.load(f)
                    existing_posts = existing_data.get('posts', [])
            except Exception as e:
                logger.warning(f"Could not load existing posts: {e}")
        
        # Combine with new posts
        all_posts = existing_posts + posts
        
        posts_data = {
            'posts': all_posts,
            'total_count': len(all_posts),
            'generated_at': datetime.now().isoformat(),
            'sources': ['twitter', 'facebook', 'instagram', 'tiktok']
        }
        
        with open(posts_file, 'w') as f:
            json.dump(posts_data, f, indent=2, default=str)
        logger.info(f"Saved {len(all_posts)} social media posts to {posts_file}")
        
        # Generate and save sample resources
        resources = self.generate_sample_resources()
        resources_file = self.datasets_path / 'sample_resources.json'
        with open(resources_file, 'w') as f:
            json.dump(resources, f, indent=2, default=str)
        logger.info(f"Saved resources data to {resources_file}")
        
        # Create sample images
        self.create_sample_images()
        
        # Create summary file
        summary = {
            'data_generated_at': datetime.now().isoformat(),
            'files_created': [
                'sample_incidents.json',
                'social_media_posts.json',
                'sample_resources.json',
                'sample_images/'
            ],
            'statistics': {
                'incidents': len(incidents),
                'social_media_posts': len(all_posts),
                'fire_engines': len(resources['fire_engines']),
                'ambulances': len(resources['ambulances']),
                'police_units': len(resources['police_units'])
            },
            'usage': {
                'description': 'Sample data for testing and demonstration',
                'load_command': 'python scripts/load_demo_data.py',
                'data_location': 'data/datasets/'
            }
        }
        
        summary_file = self.datasets_path / 'data_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Created data summary: {summary_file}")
        
        return summary

async def load_sample_data_to_database():
    """Load sample data into database"""
    try:
        from models.database import get_database_manager
        from models.incident_model import Incident, ResponseTeam, create_sample_data
        
        logger.info("Loading sample data into database...")
        
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            # Create sample data using the model's built-in function
            create_sample_data(session)
            logger.info("Sample data loaded into database successfully")
            
    except ImportError:
        logger.warning("Database modules not available. Skipping database loading.")
    except Exception as e:
        logger.error(f"Failed to load sample data into database: {e}")

def main():
    """Main function to generate and load demo data"""
    logger.info("🚀 Starting demo data loading process...")
    
    try:
        # Create sample data loader
        loader = SampleDataLoader()
        
        # Generate and save sample data
        summary = loader.save_sample_data()
        
        # Load data into database
        asyncio.run(load_sample_data_to_database())
        
        logger.info("✅ Demo data loading completed successfully!")
        logger.info(f"📊 Generated {summary['statistics']['incidents']} incidents, "
                   f"{summary['statistics']['social_media_posts']} social media posts, "
                   f"and {summary['statistics']['fire_engines'] + summary['statistics']['ambulances'] + summary['statistics']['police_units']} emergency resources")
        
        print("\n" + "="*60)
        print("🎉 DEMO DATA LOADED SUCCESSFULLY!")
        print("="*60)
        print(f"📍 Data Location: data/datasets/")
        print(f"📊 Total Incidents: {summary['statistics']['incidents']}")
        print(f"📱 Social Media Posts: {summary['statistics']['social_media_posts']}")
        print(f"🚒 Emergency Resources: {summary['statistics']['fire_engines'] + summary['statistics']['ambulances'] + summary['statistics']['police_units']}")
        print(f"🖼️  Sample Images: 5 disaster scenarios")
        print("\n🚀 Your disaster response system is now ready with demo data!")
        print("💡 Start the system with: python main.py")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load demo data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)