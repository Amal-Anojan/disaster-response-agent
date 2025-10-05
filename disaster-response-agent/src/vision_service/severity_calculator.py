import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class DamageAssessment:
    """Damage assessment result"""
    severity_score: float
    confidence: float
    damage_type: str
    affected_area: float
    risk_factors: List[str]
    recommendations: List[str]

class SeverityCalculator:
    def __init__(self):
        """Initialize severity calculator with damage assessment criteria"""
        
        # Damage type weights (higher = more severe)
        self.damage_type_weights = {
            'building_collapse': 10.0,
            'structural_damage': 8.5,
            'fire_damage': 8.0,
            'flood_damage': 7.0,
            'debris_damage': 6.0,
            'vegetation_damage': 4.0,
            'vehicle_damage': 5.0,
            'infrastructure_damage': 9.0,
            'landslide': 9.5,
            'explosion_damage': 10.0
        }
        
        # Risk factor multipliers
        self.risk_multipliers = {
            'population_density': {
                'high': 1.5,
                'medium': 1.2,
                'low': 1.0
            },
            'infrastructure_criticality': {
                'critical': 1.4,
                'important': 1.2,
                'standard': 1.0
            },
            'weather_conditions': {
                'severe': 1.3,
                'moderate': 1.1,
                'mild': 1.0
            },
            'accessibility': {
                'blocked': 1.3,
                'limited': 1.1,
                'accessible': 1.0
            }
        }
        
        # Severity thresholds
        self.severity_thresholds = {
            (0.0, 2.0): 'minimal',
            (2.0, 4.0): 'minor',
            (4.0, 6.0): 'moderate',
            (6.0, 8.0): 'major',
            (8.0, 10.0): 'severe',
            (10.0, float('inf')): 'catastrophic'
        }
    
    def calculate_severity(self, 
                         vision_analysis: Dict[str, Any],
                         text_analysis: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> DamageAssessment:
        """
        Calculate comprehensive damage severity score
        
        Args:
            vision_analysis: Results from vision model
            text_analysis: Results from text analysis (optional)
            metadata: Additional metadata (location, time, etc.)
            
        Returns:
            DamageAssessment object with severity score and details
        """
        try:
            # Extract base severity from vision analysis
            base_severity = vision_analysis.get('severity', 5.0)
            
            # Extract damage types and affected area
            damage_types = vision_analysis.get('damage_types', [])
            if isinstance(damage_types, str):
                damage_types = [damage_types]
            
            # Calculate damage type severity
            damage_type_severity = self._calculate_damage_type_severity(damage_types)
            
            # Calculate area impact
            area_impact = self._calculate_area_impact(vision_analysis)
            
            # Apply risk factor multipliers
            risk_multiplier = self._calculate_risk_multiplier(vision_analysis, text_analysis, metadata)
            
            # Calculate population impact
            population_impact = self._calculate_population_impact(vision_analysis, metadata)
            
            # Calculate infrastructure impact
            infrastructure_impact = self._calculate_infrastructure_impact(vision_analysis)
            
            # Calculate temporal factors (urgency, spreading potential)
            temporal_factor = self._calculate_temporal_factor(vision_analysis, text_analysis)
            
            # Combine all factors into final severity score
            final_severity = self._combine_severity_factors(
                base_severity=base_severity,
                damage_type_severity=damage_type_severity,
                area_impact=area_impact,
                risk_multiplier=risk_multiplier,
                population_impact=population_impact,
                infrastructure_impact=infrastructure_impact,
                temporal_factor=temporal_factor
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(vision_analysis, text_analysis)
            
            # Determine primary damage type
            primary_damage_type = self._determine_primary_damage_type(damage_types)
            
            # Calculate affected area estimate
            affected_area = self._estimate_affected_area(vision_analysis, final_severity)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(vision_analysis, text_analysis, metadata)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(final_severity, primary_damage_type, risk_factors)
            
            return DamageAssessment(
                severity_score=min(final_severity, 10.0),  # Cap at 10
                confidence=confidence,
                damage_type=primary_damage_type,
                affected_area=affected_area,
                risk_factors=risk_factors,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Severity calculation failed: {e}")
            return self._create_fallback_assessment(vision_analysis)
    
    def _calculate_damage_type_severity(self, damage_types: List[str]) -> float:
        """Calculate severity based on damage types"""
        if not damage_types:
            return 5.0  # Default severity
        
        # Get severity for each damage type
        severities = []
        for damage_type in damage_types:
            # Normalize damage type name
            normalized_type = damage_type.lower().replace(' ', '_')
            
            # Find best match in damage type weights
            best_match = None
            best_score = 0
            
            for known_type in self.damage_type_weights:
                if normalized_type in known_type or known_type in normalized_type:
                    score = len(set(normalized_type.split('_')).intersection(set(known_type.split('_'))))
                    if score > best_score:
                        best_match = known_type
                        best_score = score
            
            if best_match:
                severities.append(self.damage_type_weights[best_match])
            else:
                severities.append(5.0)  # Default for unknown types
        
        # Return weighted average, emphasizing the most severe damage
        if len(severities) == 1:
            return severities[0]
        else:
            # Weight the highest severity more heavily
            sorted_severities = sorted(severities, reverse=True)
            weighted_severity = (
                sorted_severities[0] * 0.6 +  # Most severe gets 60% weight
                sum(sorted_severities[1:]) * 0.4 / max(len(sorted_severities) - 1, 1)  # Others share 40%
            )
            return weighted_severity
    
    def _calculate_area_impact(self, vision_analysis: Dict[str, Any]) -> float:
        """Calculate impact based on affected area"""
        # Try to extract area information from vision analysis
        affected_infrastructure = vision_analysis.get('affected_infrastructure', [])
        
        if not affected_infrastructure:
            return 1.0  # No area impact modifier
        
        # Area impact based on number and type of affected infrastructure
        area_multipliers = {
            'building': 1.2,
            'road': 1.3,
            'bridge': 1.5,
            'power_line': 1.4,
            'water_system': 1.3,
            'hospital': 2.0,
            'school': 1.7,
            'residential': 1.2,
            'commercial': 1.3,
            'industrial': 1.6
        }
        
        total_multiplier = 1.0
        for infrastructure in affected_infrastructure:
            infrastructure_lower = infrastructure.lower()
            for key, multiplier in area_multipliers.items():
                if key in infrastructure_lower:
                    total_multiplier *= multiplier
                    break
        
        # Cap the area impact to prevent extreme values
        return min(total_multiplier, 2.5)
    
    def _calculate_risk_multiplier(self, 
                                 vision_analysis: Dict[str, Any],
                                 text_analysis: Optional[Dict[str, Any]],
                                 metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate risk multiplier based on various factors"""
        multiplier = 1.0
        
        # Population density (from metadata or estimated)
        if metadata and 'population_density' in metadata:
            density = metadata['population_density']
            if density > 1000:  # High density
                multiplier *= self.risk_multipliers['population_density']['high']
            elif density > 100:  # Medium density
                multiplier *= self.risk_multipliers['population_density']['medium']
        
        # Infrastructure criticality
        affected_infrastructure = vision_analysis.get('affected_infrastructure', [])
        critical_infrastructure = ['hospital', 'power_plant', 'water_treatment', 'emergency_services']
        
        for infrastructure in affected_infrastructure:
            if any(critical in infrastructure.lower() for critical in critical_infrastructure):
                multiplier *= self.risk_multipliers['infrastructure_criticality']['critical']
                break
        
        # Weather conditions (from text analysis or metadata)
        if text_analysis:
            text_content = text_analysis.get('content', '').lower()
            if any(weather in text_content for weather in ['storm', 'heavy rain', 'strong wind', 'severe weather']):
                multiplier *= self.risk_multipliers['weather_conditions']['severe']
            elif any(weather in text_content for weather in ['rain', 'wind', 'weather']):
                multiplier *= self.risk_multipliers['weather_conditions']['moderate']
        
        # Accessibility issues
        if text_analysis:
            text_content = text_analysis.get('content', '').lower()
            if any(access in text_content for access in ['blocked', 'inaccessible', 'cut off', 'isolated']):
                multiplier *= self.risk_multipliers['accessibility']['blocked']
            elif any(access in text_content for access in ['difficult access', 'limited access']):
                multiplier *= self.risk_multipliers['accessibility']['limited']
        
        return min(multiplier, 3.0)  # Cap the risk multiplier
    
    def _calculate_population_impact(self, 
                                   vision_analysis: Dict[str, Any],
                                   metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate impact based on affected population"""
        estimated_population = vision_analysis.get('estimated_affected_population', 0)
        
        if estimated_population == 0 and metadata and 'location' in metadata:
            # Estimate population based on location type or other factors
            estimated_population = 100  # Default estimate
        
        # Population impact scaling
        if estimated_population < 10:
            return 1.0
        elif estimated_population < 100:
            return 1.1
        elif estimated_population < 1000:
            return 1.3
        elif estimated_population < 10000:
            return 1.6
        else:
            return 2.0
    
    def _calculate_infrastructure_impact(self, vision_analysis: Dict[str, Any]) -> float:
        """Calculate impact based on affected infrastructure"""
        affected_infrastructure = vision_analysis.get('affected_infrastructure', [])
        
        if not affected_infrastructure:
            return 1.0
        
        # Infrastructure importance weights
        importance_weights = {
            'hospital': 2.0,
            'emergency': 1.8,
            'power': 1.6,
            'water': 1.5,
            'communication': 1.4,
            'transportation': 1.3,
            'school': 1.2,
            'residential': 1.1
        }
        
        total_impact = 1.0
        for infrastructure in affected_infrastructure:
            infrastructure_lower = infrastructure.lower()
            for key, weight in importance_weights.items():
                if key in infrastructure_lower:
                    total_impact += (weight - 1.0) * 0.3  # Scaled contribution
                    break
        
        return min(total_impact, 2.0)
    
    def _calculate_temporal_factor(self, 
                                 vision_analysis: Dict[str, Any],
                                 text_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate temporal urgency factor"""
        urgency = vision_analysis.get('urgency', 'MEDIUM')
        
        urgency_multipliers = {
            'CRITICAL': 1.5,
            'HIGH': 1.3,
            'MEDIUM': 1.0,
            'LOW': 0.8
        }
        
        base_multiplier = urgency_multipliers.get(urgency, 1.0)
        
        # Check for spreading potential in text
        if text_analysis:
            text_content = text_analysis.get('content', '').lower()
            spreading_keywords = ['spreading', 'expanding', 'growing', 'escalating', 'worsening']
            if any(keyword in text_content for keyword in spreading_keywords):
                base_multiplier *= 1.2
        
        return base_multiplier
    
    def _combine_severity_factors(self, **factors) -> float:
        """Combine all severity factors into final score"""
        base_severity = factors['base_severity']
        damage_type_severity = factors['damage_type_severity']
        area_impact = factors['area_impact']
        risk_multiplier = factors['risk_multiplier']
        population_impact = factors['population_impact']
        infrastructure_impact = factors['infrastructure_impact']
        temporal_factor = factors['temporal_factor']
        
        # Weighted combination
        combined_severity = (
            base_severity * 0.3 +  # 30% from vision model
            damage_type_severity * 0.25 +  # 25% from damage type analysis
            (base_severity * area_impact * 0.15) +  # 15% from area impact
            (base_severity * risk_multiplier * 0.1) +  # 10% from risk factors
            (base_severity * population_impact * 0.1) +  # 10% from population impact
            (base_severity * infrastructure_impact * 0.05) +  # 5% from infrastructure
            (base_severity * temporal_factor * 0.05)  # 5% from temporal factors
        )
        
        return combined_severity
    
    def _calculate_confidence(self, 
                            vision_analysis: Dict[str, Any],
                            text_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence in severity assessment"""
        vision_confidence = vision_analysis.get('confidence', 0.7)
        
        # Start with vision confidence
        total_confidence = vision_confidence * 0.7
        
        # Add text analysis confidence if available
        if text_analysis and 'confidence' in text_analysis:
            total_confidence += text_analysis['confidence'] * 0.2
        
        # Add consistency bonus if vision and text agree
        if text_analysis:
            vision_severity = vision_analysis.get('severity', 5)
            text_urgency = text_analysis.get('urgency', 'MEDIUM')
            
            # Convert text urgency to approximate severity
            urgency_to_severity = {'LOW': 3, 'MEDIUM': 5, 'HIGH': 7, 'CRITICAL': 9}
            text_severity = urgency_to_severity.get(text_urgency, 5)
            
            # Check consistency
            severity_diff = abs(vision_severity - text_severity)
            if severity_diff < 2:
                total_confidence += 0.1  # Consistency bonus
        
        return min(total_confidence, 1.0)
    
    def _determine_primary_damage_type(self, damage_types: List[str]) -> str:
        """Determine the primary damage type"""
        if not damage_types:
            return 'unknown'
        
        # If multiple types, return the most severe one
        if len(damage_types) == 1:
            return damage_types[0]
        
        # Find the damage type with highest severity weight
        max_severity = 0
        primary_type = damage_types[0]
        
        for damage_type in damage_types:
            normalized_type = damage_type.lower().replace(' ', '_')
            
            for known_type, weight in self.damage_type_weights.items():
                if normalized_type in known_type or known_type in normalized_type:
                    if weight > max_severity:
                        max_severity = weight
                        primary_type = damage_type
                    break
        
        return primary_type
    
    def _estimate_affected_area(self, vision_analysis: Dict[str, Any], severity: float) -> float:
        """Estimate affected area in square meters"""
        # Base area estimation based on severity
        base_area = severity * 100  # Base: 100 sq meters per severity point
        
        # Adjust based on damage type
        damage_types = vision_analysis.get('damage_types', [])
        area_multipliers = {
            'flood': 5.0,
            'fire': 3.0,
            'earthquake': 4.0,
            'storm': 2.0,
            'landslide': 2.5
        }
        
        multiplier = 1.0
        for damage_type in damage_types:
            for key, mult in area_multipliers.items():
                if key in damage_type.lower():
                    multiplier = max(multiplier, mult)
                    break
        
        return base_area * multiplier
    
    def _identify_risk_factors(self, 
                             vision_analysis: Dict[str, Any],
                             text_analysis: Optional[Dict[str, Any]],
                             metadata: Optional[Dict[str, Any]]) -> List[str]:
        """Identify risk factors present in the situation"""
        risk_factors = []
        
        # Structural risks
        if 'building' in str(vision_analysis.get('affected_infrastructure', [])).lower():
            risk_factors.append('structural_instability')
        
        # Population risks
        estimated_pop = vision_analysis.get('estimated_affected_population', 0)
        if estimated_pop > 100:
            risk_factors.append('high_population_exposure')
        
        # Infrastructure risks
        affected_infra = vision_analysis.get('affected_infrastructure', [])
        critical_infra = ['power', 'water', 'hospital', 'emergency']
        if any(crit in str(affected_infra).lower() for crit in critical_infra):
            risk_factors.append('critical_infrastructure_impact')
        
        # Environmental risks
        if text_analysis:
            text_content = text_analysis.get('content', '').lower()
            env_risks = {
                'secondary_collapse': ['aftershock', 'unstable', 'weakened structure'],
                'fire_spread': ['spreading fire', 'wind', 'dry conditions'],
                'flood_expansion': ['rising water', 'dam', 'levee', 'overflow'],
                'hazmat_exposure': ['chemical', 'toxic', 'gas leak', 'contamination']
            }
            
            for risk_type, keywords in env_risks.items():
                if any(keyword in text_content for keyword in keywords):
                    risk_factors.append(risk_type)
        
        # Weather-related risks
        severity = vision_analysis.get('severity', 5)
        if severity > 7:
            risk_factors.append('high_severity_impact')
        
        return list(set(risk_factors))  # Remove duplicates
    
    def _generate_recommendations(self, 
                                severity: float,
                                damage_type: str,
                                risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on assessment"""
        recommendations = []
        
        # Severity-based recommendations
        if severity >= 8:
            recommendations.append('Immediate evacuation of affected area')
            recommendations.append('Deploy maximum emergency resources')
        elif severity >= 6:
            recommendations.append('Establish safety perimeter')
            recommendations.append('Deploy specialized response teams')
        elif severity >= 4:
            recommendations.append('Monitor situation closely')
            recommendations.append('Prepare additional resources')
        
        # Damage type specific recommendations
        damage_recommendations = {
            'fire': ['Deploy fire suppression units', 'Establish firebreaks'],
            'flood': ['Deploy water rescue teams', 'Set up evacuation shelters'],
            'earthquake': ['Deploy search and rescue teams', 'Assess structural integrity'],
            'collapse': ['Deploy urban search and rescue', 'Establish exclusion zone'],
            'landslide': ['Evacuate downslope areas', 'Monitor slope stability']
        }
        
        for damage_key, recs in damage_recommendations.items():
            if damage_key in damage_type.lower():
                recommendations.extend(recs)
                break
        
        # Risk factor specific recommendations
        risk_recommendations = {
            'structural_instability': 'Deploy structural engineers for assessment',
            'high_population_exposure': 'Implement mass evacuation procedures',
            'critical_infrastructure_impact': 'Prioritize infrastructure protection',
            'secondary_collapse': 'Establish wide safety perimeter',
            'fire_spread': 'Deploy additional fire suppression resources',
            'flood_expansion': 'Monitor water levels and evacuation routes',
            'hazmat_exposure': 'Deploy hazmat teams and decontamination units'
        }
        
        for risk_factor in risk_factors:
            if risk_factor in risk_recommendations:
                recommendations.append(risk_recommendations[risk_factor])
        
        # Remove duplicates and return top recommendations
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:6]  # Return top 6 recommendations
    
    def _create_fallback_assessment(self, vision_analysis: Dict[str, Any]) -> DamageAssessment:
        """Create fallback assessment when calculation fails"""
        base_severity = vision_analysis.get('severity', 5.0)
        
        return DamageAssessment(
            severity_score=base_severity,
            confidence=0.5,
            damage_type=vision_analysis.get('disaster_type', 'unknown'),
            affected_area=base_severity * 100,
            risk_factors=['assessment_uncertainty'],
            recommendations=['Deploy emergency assessment team', 'Gather additional information']
        )

# Usage example
def test_severity_calculator():
    """Test the severity calculator"""
    calculator = SeverityCalculator()
    
    # Sample vision analysis
    vision_analysis = {
        'severity': 7,
        'damage_types': ['building_collapse', 'fire_damage'],
        'affected_infrastructure': ['residential_buildings', 'power_lines'],
        'estimated_affected_population': 500,
        'urgency': 'HIGH',
        'confidence': 0.85
    }
    
    # Sample text analysis
    text_analysis = {
        'content': 'Major building collapse with fire spreading rapidly',
        'urgency': 'CRITICAL',
        'confidence': 0.8
    }
    
    # Calculate severity
    assessment = calculator.calculate_severity(vision_analysis, text_analysis)
    
    print(f"Severity Score: {assessment.severity_score:.2f}")
    print(f"Confidence: {assessment.confidence:.2f}")
    print(f"Primary Damage Type: {assessment.damage_type}")
    print(f"Affected Area: {assessment.affected_area:.0f} sq meters")
    print(f"Risk Factors: {assessment.risk_factors}")
    print(f"Recommendations: {assessment.recommendations}")

if __name__ == "__main__":
    test_severity_calculator()