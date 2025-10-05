import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import json
import time

logger = logging.getLogger(__name__)

class EmergencyPipeline:
    def __init__(self, 
                 vision_analyzer=None, 
                 text_analyzer=None,
                 action_generator=None,
                 rag_system=None):
        """
        Initialize disaster response processing pipeline
        
        Args:
            vision_analyzer: Vision analysis service
            text_analyzer: Text analysis service  
            action_generator: Action plan generation service
            rag_system: RAG system for knowledge retrieval
        """
        self.vision_analyzer = vision_analyzer
        self.text_analyzer = text_analyzer
        self.action_generator = action_generator
        self.rag_system = rag_system
        
        # Processing statistics
        self.processing_stats = {
            'total_incidents': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'average_processing_time': 0.0
        }
        
        # Processing queue for batch operations
        self.processing_queue = asyncio.Queue()
        self.active_workers = 0
        self.max_workers = 5
    
    async def process_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete emergency incident through the pipeline
        
        Args:
            incident_data: Dictionary containing incident information
                - incident_id: Unique identifier
                - text_content: Text description (optional)
                - image_data: Image bytes (optional) 
                - location: GPS coordinates (optional)
                - metadata: Additional metadata (optional)
                
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        incident_id = incident_data.get('incident_id', str(uuid.uuid4()))
        
        try:
            logger.info(f"Processing incident {incident_id}")
            
            # Initialize result structure
            result = {
                'incident_id': incident_id,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_status': 'PROCESSING',
                'vision_analysis': {},
                'text_analysis': {},
                'combined_analysis': {},
                'action_plan': {},
                'processing_time_seconds': 0.0,
                'confidence_score': 0.0,
                'errors': []
            }
            
            # Stage 1: Vision Analysis (if image provided)
            if incident_data.get('image_data'):
                try:
                    logger.debug(f"Starting vision analysis for {incident_id}")
                    vision_result = await self._process_vision_analysis(
                        incident_data['image_data'],
                        incident_data.get('metadata', {})
                    )
                    result['vision_analysis'] = vision_result
                    logger.debug(f"Vision analysis completed for {incident_id}")
                except Exception as e:
                    error_msg = f"Vision analysis failed: {str(e)}"
                    logger.error(error_msg)
                    result['errors'].append(error_msg)
                    result['vision_analysis'] = self._create_fallback_vision_analysis()
            
            # Stage 2: Text Analysis (if text provided)
            if incident_data.get('text_content'):
                try:
                    logger.debug(f"Starting text analysis for {incident_id}")
                    text_result = await self._process_text_analysis(
                        incident_data['text_content'],
                        incident_data.get('metadata', {})
                    )
                    result['text_analysis'] = text_result
                    logger.debug(f"Text analysis completed for {incident_id}")
                except Exception as e:
                    error_msg = f"Text analysis failed: {str(e)}"
                    logger.error(error_msg)
                    result['errors'].append(error_msg)
                    result['text_analysis'] = self._create_fallback_text_analysis(
                        incident_data.get('text_content', '')
                    )
            
            # Stage 3: Combine Analysis Results
            try:
                logger.debug(f"Combining analysis results for {incident_id}")
                combined_analysis = await self._combine_analysis_results(
                    result['vision_analysis'],
                    result['text_analysis'],
                    incident_data
                )
                result['combined_analysis'] = combined_analysis
                logger.debug(f"Analysis combination completed for {incident_id}")
            except Exception as e:
                error_msg = f"Analysis combination failed: {str(e)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                result['combined_analysis'] = self._create_fallback_combined_analysis(
                    result['vision_analysis'], 
                    result['text_analysis']
                )
            
            # Stage 4: Generate Action Plan
            try:
                logger.debug(f"Generating action plan for {incident_id}")
                action_plan = await self._generate_action_plan(
                    result['combined_analysis'],
                    incident_data.get('text_content', ''),
                    incident_data.get('location')
                )
                result['action_plan'] = action_plan
                logger.debug(f"Action plan generated for {incident_id}")
            except Exception as e:
                error_msg = f"Action plan generation failed: {str(e)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                result['action_plan'] = self._create_fallback_action_plan(
                    result['combined_analysis']
                )
            
            # Calculate final metrics
            processing_time = time.time() - start_time
            result['processing_time_seconds'] = processing_time
            result['confidence_score'] = self._calculate_overall_confidence(result)
            result['processing_status'] = 'COMPLETED' if not result['errors'] else 'COMPLETED_WITH_ERRORS'
            
            # Update statistics
            self._update_processing_stats(processing_time, len(result['errors']) == 0)
            
            logger.info(f"Incident {incident_id} processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Pipeline processing failed: {str(e)}"
            logger.error(error_msg)
            
            # Update statistics for failure
            self._update_processing_stats(processing_time, False)
            
            return {
                'incident_id': incident_id,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_status': 'FAILED',
                'processing_time_seconds': processing_time,
                'error': error_msg,
                'confidence_score': 0.0
            }
    
    async def _process_vision_analysis(self, image_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process image through vision analysis"""
        if not self.vision_analyzer:
            raise ValueError("Vision analyzer not available")
        
        # Call vision analyzer
        vision_result = await self.vision_analyzer.analyze_disaster_image(image_data, metadata)
        
        # Add processing metadata
        vision_result['processing_method'] = 'vision_ai'
        vision_result['processing_timestamp'] = datetime.now().isoformat()
        
        return vision_result
    
    async def _process_text_analysis(self, text_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process text through text analysis"""
        if not self.text_analyzer:
            raise ValueError("Text analyzer not available")
        
        # Call text analyzer
        text_result = await self.text_analyzer.analyze_text(text_content, metadata)
        
        # Add processing metadata
        text_result['processing_method'] = 'text_nlp'
        text_result['processing_timestamp'] = datetime.now().isoformat()
        
        return text_result
    
    async def _combine_analysis_results(self, 
                                      vision_analysis: Dict[str, Any],
                                      text_analysis: Dict[str, Any],
                                      incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine vision and text analysis results"""
        
        combined_result = {
            'severity_score': 5.0,
            'confidence_score': 0.5,
            'disaster_type': 'unknown',
            'urgency_level': 'MEDIUM',
            'affected_infrastructure': [],
            'estimated_affected_population': 0,
            'key_findings': [],
            'risk_factors': [],
            'location_info': {},
            'temporal_factors': {}
        }
        
        # Extract severity from both sources
        vision_severity = vision_analysis.get('severity', 5.0)
        text_urgency = text_analysis.get('urgency_level', 'MEDIUM')
        
        # Convert text urgency to severity scale
        urgency_to_severity = {
            'LOW': 3.0,
            'MEDIUM': 5.0, 
            'HIGH': 7.0,
            'CRITICAL': 9.0
        }
        text_severity = urgency_to_severity.get(text_urgency, 5.0)
        
        # Combine severity scores (weighted average)
        if vision_analysis and text_analysis:
            # Both sources available
            combined_severity = (vision_severity * 0.6) + (text_severity * 0.4)
        elif vision_analysis:
            # Only vision available
            combined_severity = vision_severity
        elif text_analysis:
            # Only text available
            combined_severity = text_severity
        else:
            # Neither available (shouldn't happen)
            combined_severity = 5.0
        
        combined_result['severity_score'] = min(max(combined_severity, 1.0), 10.0)
        
        # Determine disaster type
        vision_disaster_type = vision_analysis.get('disaster_type', '')
        text_disaster_types = text_analysis.get('disaster_types', [])
        
        if vision_disaster_type and text_disaster_types:
            # Check for consistency
            if vision_disaster_type in text_disaster_types:
                combined_result['disaster_type'] = vision_disaster_type
            else:
                # Use the one with higher confidence
                vision_confidence = vision_analysis.get('confidence', 0.5)
                text_confidence = text_analysis.get('confidence', 0.5)
                
                if vision_confidence > text_confidence:
                    combined_result['disaster_type'] = vision_disaster_type
                else:
                    combined_result['disaster_type'] = text_disaster_types[0] if text_disaster_types else 'unknown'
        elif vision_disaster_type:
            combined_result['disaster_type'] = vision_disaster_type
        elif text_disaster_types:
            combined_result['disaster_type'] = text_disaster_types[0]
        
        # Combine urgency level
        urgency_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        vision_urgency = vision_analysis.get('urgency', 'MEDIUM')
        text_urgency_level = text_analysis.get('urgency_level', 'MEDIUM')
        
        # Take the higher urgency level
        vision_urgency_idx = urgency_levels.index(vision_urgency) if vision_urgency in urgency_levels else 1
        text_urgency_idx = urgency_levels.index(text_urgency_level) if text_urgency_level in urgency_levels else 1
        
        combined_urgency_idx = max(vision_urgency_idx, text_urgency_idx)
        combined_result['urgency_level'] = urgency_levels[combined_urgency_idx]
        
        # Combine affected infrastructure
        vision_infrastructure = vision_analysis.get('affected_infrastructure', [])
        
        if isinstance(vision_infrastructure, list):
            combined_result['affected_infrastructure'].extend(vision_infrastructure)
        
        # Add infrastructure from text locations
        text_locations = text_analysis.get('locations', [])
        for location in text_locations:
            if location.get('type') in ['medical_facility', 'educational', 'commercial', 'infrastructure']:
                combined_result['affected_infrastructure'].append(location.get('text', 'unknown'))
        
        # Remove duplicates
        combined_result['affected_infrastructure'] = list(set(combined_result['affected_infrastructure']))
        
        # Estimate affected population
        vision_population = vision_analysis.get('estimated_affected_population', 0)
        text_numbers = text_analysis.get('numbers', {})
        text_population = sum(text_numbers.get('affected_people', [0])) + sum(text_numbers.get('casualties', [0]))
        
        combined_result['estimated_affected_population'] = max(vision_population, text_population)
        
        # Combine key findings
        key_findings = []
        
        if vision_analysis.get('hazards'):
            key_findings.extend(vision_analysis['hazards'])
        
        if text_analysis.get('key_phrases'):
            key_findings.extend(text_analysis['key_phrases'][:3])  # Top 3 phrases
        
        combined_result['key_findings'] = key_findings
        
        # Location information
        if incident_data.get('location'):
            combined_result['location_info'] = incident_data['location']
        elif text_analysis.get('locations'):
            # Use first location found in text
            first_location = text_analysis['locations'][0]
            combined_result['location_info'] = {
                'description': first_location.get('text', ''),
                'type': first_location.get('type', 'general')
            }
        
        # Temporal factors
        if text_analysis.get('temporal_info'):
            combined_result['temporal_factors'] = text_analysis['temporal_info']
        
        # Calculate combined confidence
        vision_confidence = vision_analysis.get('confidence', 0.5)
        text_confidence = text_analysis.get('confidence', 0.5)
        
        if vision_analysis and text_analysis:
            # Both available - weighted average with consistency bonus
            base_confidence = (vision_confidence * 0.6) + (text_confidence * 0.4)
            
            # Consistency bonus if vision and text agree on disaster type
            if vision_disaster_type and vision_disaster_type in text_disaster_types:
                base_confidence += 0.1
            
            combined_result['confidence_score'] = min(base_confidence, 1.0)
        elif vision_analysis:
            combined_result['confidence_score'] = vision_confidence
        elif text_analysis:
            combined_result['confidence_score'] = text_confidence
        else:
            combined_result['confidence_score'] = 0.3
        
        return combined_result
    
    async def _generate_action_plan(self, 
                                  combined_analysis: Dict[str, Any],
                                  text_content: str,
                                  location: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Generate action plan using combined analysis"""
        if not self.action_generator:
            raise ValueError("Action generator not available")
        
        # Generate action plan
        action_plan = await self.action_generator.generate_action_plan(
            combined_analysis,
            text_content,
            location
        )
        
        # Add processing metadata
        action_plan['generation_method'] = 'llm_rag'
        action_plan['generation_timestamp'] = datetime.now().isoformat()
        
        return action_plan
    
    def _calculate_overall_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the processing result"""
        confidence_scores = []
        
        # Vision analysis confidence
        if result.get('vision_analysis', {}).get('confidence'):
            confidence_scores.append(result['vision_analysis']['confidence'])
        
        # Text analysis confidence  
        if result.get('text_analysis', {}).get('confidence'):
            confidence_scores.append(result['text_analysis']['confidence'])
        
        # Combined analysis confidence
        if result.get('combined_analysis', {}).get('confidence_score'):
            confidence_scores.append(result['combined_analysis']['confidence_score'])
        
        # Action plan confidence (if available)
        if result.get('action_plan', {}).get('confidence'):
            confidence_scores.append(result['action_plan']['confidence'])
        
        # Error penalty
        error_count = len(result.get('errors', []))
        error_penalty = min(error_count * 0.1, 0.5)  # Max 50% penalty
        
        # Calculate weighted average
        if confidence_scores:
            base_confidence = sum(confidence_scores) / len(confidence_scores)
            return max(base_confidence - error_penalty, 0.0)
        else:
            return max(0.3 - error_penalty, 0.0)  # Base confidence minus penalty
    
    def _update_processing_stats(self, processing_time: float, success: bool):
        """Update processing statistics"""
        self.processing_stats['total_incidents'] += 1
        
        if success:
            self.processing_stats['successful_processing'] += 1
        else:
            self.processing_stats['failed_processing'] += 1
        
        # Update average processing time
        total_incidents = self.processing_stats['total_incidents']
        current_avg = self.processing_stats['average_processing_time']
        
        # Running average calculation
        self.processing_stats['average_processing_time'] = (
            (current_avg * (total_incidents - 1) + processing_time) / total_incidents
        )
    
    def _create_fallback_vision_analysis(self) -> Dict[str, Any]:
        """Create fallback vision analysis when vision processing fails"""
        return {
            'severity': 5.0,
            'disaster_type': 'unknown',
            'urgency': 'MEDIUM',
            'confidence': 0.3,
            'processing_method': 'fallback',
            'note': 'Vision analysis unavailable'
        }
    
    def _create_fallback_text_analysis(self, text_content: str) -> Dict[str, Any]:
        """Create fallback text analysis when text processing fails"""
        return {
            'disaster_types': ['unknown'],
            'primary_disaster_type': 'unknown',
            'urgency_level': 'MEDIUM',
            'confidence': 0.3,
            'original_text': text_content,
            'processing_method': 'fallback',
            'note': 'Text analysis unavailable'
        }
    
    def _create_fallback_combined_analysis(self, 
                                         vision_analysis: Dict[str, Any],
                                         text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback combined analysis"""
        return {
            'severity_score': 5.0,
            'confidence_score': 0.3,
            'disaster_type': 'unknown',
            'urgency_level': 'MEDIUM',
            'affected_infrastructure': [],
            'estimated_affected_population': 0,
            'processing_method': 'fallback',
            'note': 'Analysis combination unavailable'
        }
    
    def _create_fallback_action_plan(self, combined_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback action plan when generation fails"""
        disaster_type = combined_analysis.get('disaster_type', 'unknown')
        severity = combined_analysis.get('severity_score', 5.0)
        
        return {
            'immediate_actions': [
                f"Deploy {disaster_type} response teams",
                "Establish incident command post",
                "Assess immediate safety hazards"
            ],
            'resource_requirements': {
                'personnel': {'emergency_responders': max(3, int(severity))},
                'equipment': ['basic_emergency_equipment'],
                'vehicles': ['emergency_vehicles']
            },
            'metrics': {
                'estimated_response_time_minutes': max(10, 20 - int(severity)),
                'total_personnel_required': max(5, int(severity * 2))
            },
            'processing_method': 'fallback',
            'note': 'Action plan generation unavailable'
        }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self.processing_stats,
            'success_rate': (
                self.processing_stats['successful_processing'] / 
                max(self.processing_stats['total_incidents'], 1)
            ),
            'active_workers': self.active_workers
        }
    
    async def process_batch(self, incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple incidents in batch"""
        logger.info(f"Processing batch of {len(incidents)} incidents")
        
        # Create processing tasks
        tasks = []
        for incident in incidents:
            task = asyncio.create_task(self.process_incident(incident))
            tasks.append(task)
        
        # Process all incidents concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = {
                    'incident_id': incidents[i].get('incident_id', f'batch_{i}'),
                    'processing_status': 'FAILED',
                    'error': str(result),
                    'processing_timestamp': datetime.now().isoformat()
                }
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        logger.info(f"Batch processing completed: {len(processed_results)} results")
        return processed_results

# Usage example and testing
async def test_pipeline():
    """Test the disaster pipeline"""
    # This would normally import the actual services
    # from vision_service.damage_analyzer import CerebrasVisionAnalyzer
    # from llm_service.action_generator import EmergencyActionGenerator
    # from llm_service.text_analyzer import EmergencyTextAnalyzer
    # from llm_service.rag_system import EmergencyRAGSystem
    
    print("Testing Disaster Pipeline (with mock services)")
    
    # Create pipeline (normally with real services)
    pipeline = EmergencyPipeline()
    
    # Test incident
    test_incident = {
        'incident_id': 'TEST_001',
        'text_content': 'Major apartment fire on Main Street, people trapped on upper floors',
        'image_data': None,  # Would be actual image bytes
        'location': {'lat': 37.7749, 'lng': -122.4194},
        'metadata': {'source': 'test', 'timestamp': datetime.now().isoformat()}
    }
    
    # Process incident
    result = await pipeline.process_incident(test_incident)
    
    print(f"Processing Status: {result['processing_status']}")
    print(f"Processing Time: {result['processing_time_seconds']:.2f}s")
    print(f"Confidence Score: {result['confidence_score']:.2f}")
    print(f"Errors: {len(result.get('errors', []))}")
    
    # Get statistics
    stats = pipeline.get_processing_statistics()
    print(f"Pipeline Statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(test_pipeline())