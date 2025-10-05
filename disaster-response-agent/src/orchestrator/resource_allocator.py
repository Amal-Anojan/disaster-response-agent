import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math
import json

logger = logging.getLogger(__name__)

class ResourceOptimizer:
    def __init__(self):
        """Initialize resource allocation optimizer"""
        
        # Resource types and their capabilities
        self.resource_types = {
            'fire_engine': {
                'capacity': 6,  # Personnel capacity
                'response_time_factor': 1.0,
                'specialization': ['fire_suppression', 'rescue'],
                'cost_per_hour': 800
            },
            'ambulance': {
                'capacity': 3,
                'response_time_factor': 0.9,  # Faster response
                'specialization': ['medical', 'transport'],
                'cost_per_hour': 600
            },
            'police_unit': {
                'capacity': 2,
                'response_time_factor': 0.8,  # Fastest response
                'specialization': ['security', 'traffic_control'],
                'cost_per_hour': 400
            },
            'rescue_vehicle': {
                'capacity': 8,
                'response_time_factor': 1.2,
                'specialization': ['search_rescue', 'heavy_rescue'],
                'cost_per_hour': 1000
            },
            'hazmat_unit': {
                'capacity': 4,
                'response_time_factor': 1.5,
                'specialization': ['hazmat', 'decontamination'],
                'cost_per_hour': 1200
            }
        }
        
        # Personnel specializations
        self.personnel_specializations = {
            'firefighter': ['fire_suppression', 'basic_rescue', 'hazmat_level_1'],
            'paramedic': ['advanced_medical', 'triage', 'transport'],
            'police_officer': ['security', 'crowd_control', 'traffic_management'],
            'rescue_specialist': ['search_rescue', 'confined_space', 'rope_rescue'],
            'hazmat_technician': ['chemical_response', 'decontamination', 'air_monitoring'],
            'incident_commander': ['coordination', 'resource_management', 'communications']
        }
        
        # Equipment requirements by disaster type
        self.equipment_requirements = {
            'fire': ['hoses', 'breathing_apparatus', 'thermal_cameras', 'ladders'],
            'flood': ['boats', 'life_jackets', 'water_pumps', 'sandbags'],
            'earthquake': ['search_cameras', 'cutting_tools', 'lifting_equipment', 'medical_supplies'],
            'hazmat': ['protective_suits', 'detection_equipment', 'decontamination_supplies'],
            'storm': ['chainsaws', 'generators', 'emergency_lighting', 'tarps']
        }

class EmergencyResourceAllocator:
    def __init__(self):
        """Initialize emergency resource allocator"""
        self.optimizer = ResourceOptimizer()
        
        # Simulated available resources (in production, this would come from a database)
        self.available_resources = self._initialize_available_resources()
        
        # Resource allocation history
        self.allocation_history = []
        
        # Performance metrics
        self.allocation_metrics = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'average_response_time': 0.0,
            'resource_utilization_rate': 0.0
        }
    
    def _initialize_available_resources(self) -> Dict[str, Any]:
        """Initialize available resources (mock data for demo)"""
        return {
            'fire_engines': [
                {'id': 'FE001', 'location': {'lat': 37.7749, 'lng': -122.4194}, 'status': 'available', 'crew_size': 6},
                {'id': 'FE002', 'location': {'lat': 37.7849, 'lng': -122.4094}, 'status': 'available', 'crew_size': 6},
                {'id': 'FE003', 'location': {'lat': 37.7649, 'lng': -122.4294}, 'status': 'deployed', 'crew_size': 6},
            ],
            'ambulances': [
                {'id': 'AMB001', 'location': {'lat': 37.7749, 'lng': -122.4194}, 'status': 'available', 'crew_size': 3},
                {'id': 'AMB002', 'location': {'lat': 37.7849, 'lng': -122.4094}, 'status': 'available', 'crew_size': 3},
                {'id': 'AMB003', 'location': {'lat': 37.7649, 'lng': -122.4294}, 'status': 'maintenance', 'crew_size': 3},
                {'id': 'AMB004', 'location': {'lat': 37.7550, 'lng': -122.4400}, 'status': 'available', 'crew_size': 3},
            ],
            'police_units': [
                {'id': 'PU001', 'location': {'lat': 37.7749, 'lng': -122.4194}, 'status': 'available', 'crew_size': 2},
                {'id': 'PU002', 'location': {'lat': 37.7849, 'lng': -122.4094}, 'status': 'available', 'crew_size': 2},
                {'id': 'PU003', 'location': {'lat': 37.7649, 'lng': -122.4294}, 'status': 'available', 'crew_size': 2},
                {'id': 'PU004', 'location': {'lat': 37.7550, 'lng': -122.4400}, 'status': 'deployed', 'crew_size': 2},
            ],
            'rescue_vehicles': [
                {'id': 'RV001', 'location': {'lat': 37.7749, 'lng': -122.4194}, 'status': 'available', 'crew_size': 8},
                {'id': 'RV002', 'location': {'lat': 37.7849, 'lng': -122.4094}, 'status': 'available', 'crew_size': 8},
            ],
            'hazmat_units': [
                {'id': 'HZ001', 'location': {'lat': 37.7749, 'lng': -122.4194}, 'status': 'available', 'crew_size': 4},
            ]
        }
    
    async def allocate_resources(self, 
                               incident_analysis: Dict[str, Any],
                               location: Optional[Dict[str, float]] = None,
                               constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Allocate optimal resources for emergency incident
        
        Args:
            incident_analysis: Analysis results from the processing pipeline
            location: Incident location coordinates
            constraints: Resource allocation constraints
            
        Returns:
            Resource allocation plan
        """
        try:
            allocation_start_time = datetime.now()
            
            # Extract key parameters from incident analysis
            disaster_type = incident_analysis.get('disaster_type', 'unknown')
            severity = incident_analysis.get('severity_score', 5.0)
            urgency = incident_analysis.get('urgency_level', 'MEDIUM')
            affected_population = incident_analysis.get('estimated_affected_population', 100)
            
            # Calculate resource requirements
            resource_requirements = self._calculate_resource_requirements(
                disaster_type, severity, affected_population, urgency
            )
            
            # Find available resources within response radius
            response_radius = self._determine_response_radius(urgency, severity)
            available_resources = self._get_available_resources_in_radius(
                location, response_radius
            )
            
            # Optimize resource allocation
            allocation_plan = await self._optimize_resource_allocation(
                resource_requirements,
                available_resources,
                location,
                constraints or {}
            )
            
            # Calculate response metrics
            response_metrics = self._calculate_response_metrics(
                allocation_plan, location, severity
            )
            
            # Generate backup plan
            backup_plan = self._generate_backup_plan(
                resource_requirements, allocation_plan
            )
            
            # Create final allocation result
            allocation_result = {
                'allocation_id': f"ALLOC_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'incident_analysis': {
                    'disaster_type': disaster_type,
                    'severity': severity,
                    'urgency': urgency,
                    'affected_population': affected_population
                },
                'resource_requirements': resource_requirements,
                'allocated_resources': allocation_plan,
                'backup_resources': backup_plan,
                'response_metrics': response_metrics,
                'allocation_status': 'SUCCESS',
                'allocation_timestamp': allocation_start_time.isoformat(),
                'constraints_applied': constraints or {}
            }
            
            # Record allocation
            self._record_allocation(allocation_result)
            
            return allocation_result
            
        except Exception as e:
            logger.error(f"Resource allocation failed: {e}")
            return self._create_fallback_allocation(incident_analysis, str(e))
    
    def _calculate_resource_requirements(self, 
                                       disaster_type: str,
                                       severity: float,
                                       affected_population: int,
                                       urgency: str) -> Dict[str, Any]:
        """Calculate required resources based on incident parameters"""
        
        # Base requirements by disaster type
        base_requirements = {
            'fire': {
                'fire_engines': max(2, math.ceil(severity / 3)),
                'ambulances': max(1, math.ceil(severity / 4)),
                'police_units': 1,
                'personnel_count': max(12, int(severity * 3))
            },
            'flood': {
                'rescue_vehicles': max(2, math.ceil(severity / 3)),
                'ambulances': max(1, math.ceil(severity / 5)),
                'police_units': max(1, math.ceil(severity / 6)),
                'personnel_count': max(10, int(severity * 2.5))
            },
            'earthquake': {
                'rescue_vehicles': max(3, math.ceil(severity / 2)),
                'ambulances': max(2, math.ceil(severity / 3)),
                'police_units': max(2, math.ceil(severity / 4)),
                'personnel_count': max(20, int(severity * 4))
            },
            'hazmat': {
                'hazmat_units': max(1, math.ceil(severity / 4)),
                'ambulances': max(1, math.ceil(severity / 5)),
                'police_units': max(2, math.ceil(severity / 3)),
                'personnel_count': max(8, int(severity * 2))
            },
            'accident': {
                'ambulances': max(1, math.ceil(severity / 4)),
                'police_units': max(1, math.ceil(severity / 5)),
                'fire_engines': max(1, math.ceil(severity / 6)) if severity > 5 else 0,
                'personnel_count': max(6, int(severity * 2))
            }
        }
        
        # Get base requirements for disaster type
        requirements = base_requirements.get(disaster_type, {
            'rescue_vehicles': 1,
            'ambulances': 1,
            'police_units': 1,
            'personnel_count': 8
        })
        
        # Apply population scaling
        population_multiplier = self._calculate_population_multiplier(affected_population)
        for resource_type, count in requirements.items():
            if resource_type != 'personnel_count':
                requirements[resource_type] = math.ceil(count * population_multiplier)
        
        # Apply urgency scaling
        urgency_multiplier = {
            'LOW': 0.8,
            'MEDIUM': 1.0,
            'HIGH': 1.3,
            'CRITICAL': 1.6
        }.get(urgency, 1.0)
        
        for resource_type, count in requirements.items():
            requirements[resource_type] = math.ceil(count * urgency_multiplier)
        
        # Add equipment requirements
        requirements['equipment'] = self.optimizer.equipment_requirements.get(disaster_type, [])
        
        # Add specialized personnel requirements
        requirements['specialized_personnel'] = self._get_specialized_personnel_requirements(
            disaster_type, severity
        )
        
        return requirements
    
    def _calculate_population_multiplier(self, affected_population: int) -> float:
        """Calculate resource multiplier based on affected population"""
        if affected_population < 50:
            return 1.0
        elif affected_population < 200:
            return 1.2
        elif affected_population < 500:
            return 1.5
        elif affected_population < 1000:
            return 2.0
        else:
            return min(affected_population / 500, 5.0)  # Cap at 5x multiplier
    
    def _get_specialized_personnel_requirements(self, disaster_type: str, severity: float) -> List[str]:
        """Get specialized personnel requirements"""
        specializations = {
            'fire': ['firefighter', 'paramedic'] + (['hazmat_technician'] if severity > 7 else []),
            'flood': ['rescue_specialist', 'paramedic'],
            'earthquake': ['rescue_specialist', 'paramedic'] + (['incident_commander'] if severity > 7 else []),
            'hazmat': ['hazmat_technician', 'paramedic'],
            'accident': ['paramedic'] + (['firefighter'] if severity > 5 else [])
        }
        
        return specializations.get(disaster_type, ['firefighter', 'paramedic'])
    
    def _determine_response_radius(self, urgency: str, severity: float) -> float:
        """Determine search radius for resources based on urgency and severity"""
        base_radius = {
            'LOW': 10.0,      # 10 km
            'MEDIUM': 15.0,   # 15 km  
            'HIGH': 25.0,     # 25 km
            'CRITICAL': 40.0  # 40 km
        }.get(urgency, 15.0)
        
        # Increase radius for higher severity
        severity_modifier = 1.0 + (severity - 5.0) * 0.1
        
        return base_radius * severity_modifier
    
    def _get_available_resources_in_radius(self, 
                                         location: Optional[Dict[str, float]],
                                         radius_km: float) -> Dict[str, List[Dict[str, Any]]]:
        """Get available resources within specified radius"""
        if not location:
            # If no location specified, return all available resources
            return {
                resource_type: [r for r in resources if r['status'] == 'available']
                for resource_type, resources in self.available_resources.items()
            }
        
        available_in_radius = {}
        
        for resource_type, resources in self.available_resources.items():
            available_resources = []
            
            for resource in resources:
                if resource['status'] == 'available':
                    # Calculate distance
                    distance = self._calculate_distance(
                        location, resource['location']
                    )
                    
                    if distance <= radius_km:
                        resource_with_distance = resource.copy()
                        resource_with_distance['distance_km'] = distance
                        available_resources.append(resource_with_distance)
            
            # Sort by distance (closest first)
            available_resources.sort(key=lambda x: x['distance_km'])
            available_in_radius[resource_type] = available_resources
        
        return available_in_radius
    
    def _calculate_distance(self, loc1: Dict[str, float], loc2: Dict[str, float]) -> float:
        """Calculate distance between two points using Haversine formula"""
        lat1, lng1 = loc1['lat'], loc1['lng']
        lat2, lng2 = loc2['lat'], loc2['lng']
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        earth_radius = 6371
        
        return earth_radius * c
    
    async def _optimize_resource_allocation(self,
                                          requirements: Dict[str, Any],
                                          available_resources: Dict[str, List[Dict[str, Any]]],
                                          location: Optional[Dict[str, float]],
                                          constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize resource allocation using greedy algorithm with constraints"""
        
        allocated_resources = []
        
        # Sort resource types by priority
        resource_priority = ['hazmat_units', 'ambulances', 'fire_engines', 'rescue_vehicles', 'police_units']
        
        for resource_type in resource_priority:
            required_count = requirements.get(resource_type, 0)
            
            if required_count > 0 and resource_type in available_resources:
                available = available_resources[resource_type]
                
                # Allocate up to required count
                for i in range(min(required_count, len(available))):
                    resource = available[i]
                    
                    # Calculate estimated response time
                    response_time = self._calculate_response_time(
                        resource, location, resource_type
                    )
                    
                    # Check constraints
                    if self._meets_constraints(resource, constraints):
                        allocation = {
                            'resource_id': resource['id'],
                            'resource_type': resource_type,
                            'crew_size': resource['crew_size'],
                            'location': resource['location'],
                            'distance_km': resource.get('distance_km', 0),
                            'estimated_response_time_minutes': response_time,
                            'allocation_priority': len(allocated_resources) + 1
                        }
                        allocated_resources.append(allocation)
                        
                        # Mark as allocated (for this calculation)
                        resource['status'] = 'allocated'
        
        return allocated_resources
    
    def _calculate_response_time(self, 
                               resource: Dict[str, Any],
                               incident_location: Optional[Dict[str, float]],
                               resource_type: str) -> float:
        """Calculate estimated response time for resource"""
        
        if not incident_location:
            return 15.0  # Default response time
        
        # Get base travel time from distance
        distance_km = resource.get('distance_km', 0)
        average_speed_kmh = 50  # Average emergency vehicle speed
        travel_time_minutes = (distance_km / average_speed_kmh) * 60
        
        # Add mobilization time
        mobilization_time = {
            'police_units': 3,      # 3 minutes
            'ambulances': 4,        # 4 minutes  
            'fire_engines': 5,      # 5 minutes
            'rescue_vehicles': 7,   # 7 minutes
            'hazmat_units': 10      # 10 minutes
        }.get(resource_type, 5)
        
        # Apply resource type response time factor
        response_factor = self.optimizer.resource_types.get(resource_type, {}).get('response_time_factor', 1.0)
        
        total_response_time = (travel_time_minutes + mobilization_time) * response_factor
        
        return round(total_response_time, 1)
    
    def _meets_constraints(self, resource: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """Check if resource meets allocation constraints"""
        
        # Maximum response time constraint
        if 'max_response_time_minutes' in constraints:
            if resource.get('estimated_response_time_minutes', 0) > constraints['max_response_time_minutes']:
                return False
        
        # Maximum distance constraint
        if 'max_distance_km' in constraints:
            if resource.get('distance_km', 0) > constraints['max_distance_km']:
                return False
        
        # Minimum crew size constraint
        if 'min_crew_size' in constraints:
            if resource.get('crew_size', 0) < constraints['min_crew_size']:
                return False
        
        return True
    
    def _calculate_response_metrics(self, 
                                  allocation_plan: List[Dict[str, Any]],
                                  location: Optional[Dict[str, float]],
                                  severity: float) -> Dict[str, Any]:
        """Calculate response metrics for allocation plan"""
        
        if not allocation_plan:
            return {
                'estimated_response_time_minutes': 30,
                'total_personnel_allocated': 0,
                'resource_count': 0,
                'coverage_percentage': 0.0,
                'cost_estimate': 0
            }
        
        # Calculate response metrics
        response_times = [r['estimated_response_time_minutes'] for r in allocation_plan]
        total_personnel = sum(r['crew_size'] for r in allocation_plan)
        
        # Estimate cost (simplified)
        total_cost = 0
        for resource in allocation_plan:
            resource_type = resource['resource_type']
            cost_per_hour = self.optimizer.resource_types.get(resource_type, {}).get('cost_per_hour', 500)
            estimated_hours = max(2, severity / 2)  # Estimate 2-5 hours based on severity
            total_cost += cost_per_hour * estimated_hours
        
        # Calculate coverage percentage (simplified)
        required_resources = max(5, severity * 2)  # Rough estimate
        coverage_percentage = min((len(allocation_plan) / required_resources) * 100, 100)
        
        return {
            'estimated_response_time_minutes': min(response_times),
            'latest_response_time_minutes': max(response_times),
            'average_response_time_minutes': sum(response_times) / len(response_times),
            'total_personnel_allocated': total_personnel,
            'resource_count': len(allocation_plan),
            'coverage_percentage': coverage_percentage,
            'estimated_cost': total_cost
        }
    
    def _generate_backup_plan(self, 
                            requirements: Dict[str, Any],
                            primary_allocation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate backup resource allocation plan"""
        
        backup_resources = []
        
        # For each resource type, find next best available resources
        allocated_ids = {r['resource_id'] for r in primary_allocation}
        
        for resource_type, resources in self.available_resources.items():
            for resource in resources:
                if (resource['status'] == 'available' and 
                    resource['id'] not in allocated_ids):
                    
                    backup_allocation = {
                        'resource_id': resource['id'],
                        'resource_type': resource_type,
                        'crew_size': resource['crew_size'],
                        'location': resource['location'],
                        'backup_priority': len(backup_resources) + 1,
                        'activation_trigger': 'primary_resource_unavailable'
                    }
                    backup_resources.append(backup_allocation)
                    
                    # Limit backup resources
                    if len(backup_resources) >= 5:
                        break
        
        return backup_resources
    
    def _record_allocation(self, allocation_result: Dict[str, Any]):
        """Record allocation in history and update metrics"""
        self.allocation_history.append(allocation_result)
        
        # Update metrics
        self.allocation_metrics['total_allocations'] += 1
        
        if allocation_result['allocation_status'] == 'SUCCESS':
            self.allocation_metrics['successful_allocations'] += 1
        
        # Update average response time
        if allocation_result.get('response_metrics', {}).get('estimated_response_time_minutes'):
            current_avg = self.allocation_metrics['average_response_time']
            new_time = allocation_result['response_metrics']['estimated_response_time_minutes']
            total_allocations = self.allocation_metrics['total_allocations']
            
            self.allocation_metrics['average_response_time'] = (
                (current_avg * (total_allocations - 1) + new_time) / total_allocations
            )
    
    def _create_fallback_allocation(self, incident_analysis: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Create fallback allocation when optimization fails"""
        
        disaster_type = incident_analysis.get('disaster_type', 'unknown')
        severity = incident_analysis.get('severity_score', 5.0)
        
        # Basic fallback allocation
        fallback_resources = []
        
        if disaster_type in ['fire', 'explosion']:
            fallback_resources.append({
                'resource_type': 'fire_engines',
                'resource_count': 2,
                'estimated_response_time_minutes': 12
            })
        
        fallback_resources.append({
            'resource_type': 'ambulances',
            'resource_count': 1,
            'estimated_response_time_minutes': 10
        })
        
        fallback_resources.append({
            'resource_type': 'police_units', 
            'resource_count': 1,
            'estimated_response_time_minutes': 8
        })
        
        return {
            'allocation_id': f"FALLBACK_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'allocation_status': 'FALLBACK',
            'error': error_msg,
            'allocated_resources': fallback_resources,
            'response_metrics': {
                'estimated_response_time_minutes': 12,
                'total_personnel_allocated': 10,
                'coverage_percentage': 60.0
            },
            'allocation_timestamp': datetime.now().isoformat()
        }
    
    def get_allocation_statistics(self) -> Dict[str, Any]:
        """Get resource allocation statistics"""
        success_rate = (
            self.allocation_metrics['successful_allocations'] / 
            max(self.allocation_metrics['total_allocations'], 1)
        )
        
        return {
            **self.allocation_metrics,
            'success_rate': success_rate,
            'recent_allocations': len([
                a for a in self.allocation_history 
                if (datetime.now() - datetime.fromisoformat(a['allocation_timestamp'])).seconds < 3600
            ])
        }
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current status of all resources"""
        status_summary = {}
        
        for resource_type, resources in self.available_resources.items():
            available_count = len([r for r in resources if r['status'] == 'available'])
            deployed_count = len([r for r in resources if r['status'] == 'deployed'])
            maintenance_count = len([r for r in resources if r['status'] == 'maintenance'])
            
            status_summary[resource_type] = {
                'total': len(resources),
                'available': available_count,
                'deployed': deployed_count,
                'maintenance': maintenance_count,
                'utilization_rate': deployed_count / len(resources) if resources else 0
            }
        
        return status_summary

# Usage example and testing
async def test_resource_allocator():
    """Test the resource allocator"""
    allocator = EmergencyResourceAllocator()
    
    # Sample incident analysis
    sample_incident = {
        'disaster_type': 'fire',
        'severity_score': 7.5,
        'urgency_level': 'HIGH',
        'estimated_affected_population': 150
    }
    
    # Sample location
    sample_location = {'lat': 37.7749, 'lng': -122.4194}
    
    # Allocate resources
    allocation_result = await allocator.allocate_resources(
        sample_incident, 
        sample_location
    )
    
    print("Resource Allocation Result:")
    print(f"Status: {allocation_result['allocation_status']}")
    print(f"Allocated Resources: {len(allocation_result['allocated_resources'])}")
    print(f"Response Time: {allocation_result['response_metrics']['estimated_response_time_minutes']} min")
    print(f"Total Personnel: {allocation_result['response_metrics']['total_personnel_allocated']}")
    print(f"Coverage: {allocation_result['response_metrics']['coverage_percentage']:.1f}%")
    
    # Get statistics
    stats = allocator.get_allocation_statistics()
    print(f"Allocation Statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(test_resource_allocator())