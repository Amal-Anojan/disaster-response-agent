import unittest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.disaster_pipeline import DisasterResponsePipeline
from orchestrator.resource_allocator import EmergencyResourceAllocator
from orchestrator.mcp_server import app

class TestOrchestrator(unittest.TestCase):
    """Test cases for orchestrator components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = DisasterResponsePipeline()
        self.resource_allocator = EmergencyResourceAllocator()
        
        # Sample incident data
        self.sample_incident = {
            'text_content': 'Building fire with people trapped',
            'image_data': b'fake_image_data',
            'location': {'lat': 37.7749, 'lng': -122.4194},
            'timestamp': datetime.now().isoformat()
        }
        
        # Sample analysis results
        self.sample_analysis = {
            'disaster_type': 'fire',
            'severity': 8.5,
            'urgency': 'HIGH',
            'estimated_affected_population': 100
        }
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertIsInstance(self.pipeline, DisasterResponsePipeline)
        self.assertTrue(hasattr(self.pipeline, 'vision_analyzer'))
        self.assertTrue(hasattr(self.pipeline, 'text_analyzer'))
        self.assertTrue(hasattr(self.pipeline, 'action_generator'))
    
    @patch('src.orchestrator.disaster_pipeline.DisasterResponsePipeline._analyze_vision')
    @patch('src.orchestrator.disaster_pipeline.DisasterResponsePipeline._analyze_text')
    @patch('src.orchestrator.disaster_pipeline.DisasterResponsePipeline._generate_action_plan')
    def test_pipeline_process_incident(self, mock_action, mock_text, mock_vision):
        """Test incident processing pipeline"""
        # Mock component responses
        mock_vision.return_value = {
            'disaster_type': 'fire',
            'severity': 8.0,
            'confidence': 0.9,
            'hazards': ['fire', 'smoke']
        }
        
        mock_text.return_value = {
            'urgency_level': 'HIGH',
            'key_phrases': ['trapped', 'fire'],
            'entities': {'people_count': 50}
        }
        
        mock_action.return_value = {
            'immediate_actions': ['Deploy fire trucks'],
            'resource_requirements': {'firefighters': 8}
        }
        
        # Process incident
        result = asyncio.run(
            self.pipeline.process_incident(self.sample_incident)
        )
        
        # Verify result structure
        self.assertIn('incident_analysis', result)
        self.assertIn('processing_metadata', result)
        self.assertIn('status', result)
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling"""
        # Test with invalid incident data
        invalid_incident = {}
        
        result = asyncio.run(
            self.pipeline.process_incident(invalid_incident)
        )
        
        # Should handle gracefully
        self.assertEqual(result['status'], 'ERROR')
        self.assertIn('error', result)
    
    def test_resource_allocator_initialization(self):
        """Test resource allocator initialization"""
        self.assertIsInstance(self.resource_allocator, EmergencyResourceAllocator)
        self.assertIsInstance(self.resource_allocator.available_resources, dict)
        self.assertTrue(len(self.resource_allocator.available_resources) > 0)
    
    def test_resource_allocation_success(self):
        """Test successful resource allocation"""
        result = asyncio.run(
            self.resource_allocator.allocate_resources(
                self.sample_analysis,
                location={'lat': 37.7749, 'lng': -122.4194}
            )
        )
        
        # Verify allocation result
        self.assertEqual(result['allocation_status'], 'SUCCESS')
        self.assertIn('allocated_resources', result)
        self.assertIn('response_metrics', result)
        self.assertIsInstance(result['allocated_resources'], list)
    
    def test_resource_allocation_with_constraints(self):
        """Test resource allocation with constraints"""
        constraints = {
            'max_response_time_minutes': 10,
            'max_distance_km': 5
        }
        
        result = asyncio.run(
            self.resource_allocator.allocate_resources(
                self.sample_analysis,
                location={'lat': 37.7749, 'lng': -122.4194},
                constraints=constraints
            )
        )
        
        # Should apply constraints
        self.assertIn('constraints_applied', result)
        self.assertEqual(result['constraints_applied'], constraints)
    
    def test_priority_calculation(self):
        """Test incident priority calculation"""
        # Test high priority incident
        high_priority_incident = {
            'disaster_type': 'fire',
            'severity': 9.0,
            'urgency': 'CRITICAL',
            'estimated_affected_population': 200
        }
        
        priority = self.pipeline._calculate_incident_priority(high_priority_incident)
        self.assertGreater(priority, 15.0)  # Should be high priority
        
        # Test low priority incident  
        low_priority_incident = {
            'disaster_type': 'accident',
            'severity': 3.0,
            'urgency': 'LOW',
            'estimated_affected_population': 5
        }
        
        priority = self.pipeline._calculate_incident_priority(low_priority_incident)
        self.assertLess(priority, 10.0)  # Should be low priority
    
    def test_resource_optimization(self):
        """Test resource optimization algorithm"""
        # Create scenario with limited resources
        requirements = {
            'fire_engines': 5,
            'ambulances': 3,
            'personnel_count': 20
        }
        
        # Should optimize based on available resources
        optimization = self.resource_allocator._optimize_resource_allocation(
            requirements,
            self.resource_allocator.available_resources,
            {'lat': 37.7749, 'lng': -122.4194},
            {}
        )
        
        result = asyncio.run(optimization)
        self.assertIsInstance(result, list)
    
    def test_backup_resource_generation(self):
        """Test backup resource plan generation"""
        primary_allocation = [
            {
                'resource_id': 'FE001',
                'resource_type': 'fire_engines',
                'crew_size': 6
            }
        ]
        
        backup_plan = self.resource_allocator._generate_backup_plan(
            {'fire_engines': 2},
            primary_allocation
        )
        
        self.assertIsInstance(backup_plan, list)
        if backup_plan:
            self.assertIn('resource_id', backup_plan[0])
            self.assertIn('backup_priority', backup_plan[0])
    
    def test_response_metrics_calculation(self):
        """Test response metrics calculation"""
        allocation_plan = [
            {
                'resource_id': 'FE001',
                'estimated_response_time_minutes': 8,
                'crew_size': 6
            },
            {
                'resource_id': 'AMB001', 
                'estimated_response_time_minutes': 12,
                'crew_size': 3
            }
        ]
        
        metrics = self.resource_allocator._calculate_response_metrics(
            allocation_plan,
            {'lat': 37.7749, 'lng': -122.4194},
            7.5  # severity
        )
        
        # Verify metrics structure
        self.assertIn('estimated_response_time_minutes', metrics)
        self.assertIn('total_personnel_allocated', metrics)
        self.assertIn('resource_count', metrics)
        self.assertEqual(metrics['total_personnel_allocated'], 9)
        self.assertEqual(metrics['resource_count'], 2)
    
    def test_statistics_tracking(self):
        """Test allocation statistics tracking"""
        # Record some allocations
        for i in range(5):
            asyncio.run(
                self.resource_allocator.allocate_resources(
                    self.sample_analysis,
                    location={'lat': 37.7749, 'lng': -122.4194}
                )
            )
        
        stats = self.resource_allocator.get_allocation_statistics()
        
        # Should track statistics
        self.assertGreaterEqual(stats['total_allocations'], 5)
        self.assertIn('success_rate', stats)
        self.assertIn('average_response_time', stats)
    
    def test_resource_status_monitoring(self):
        """Test resource status monitoring"""
        status = self.resource_allocator.get_resource_status()
        
        # Should return status for all resource types
        self.assertIn('fire_engines', status)
        self.assertIn('ambulances', status)
        
        for resource_type, info in status.items():
            self.assertIn('total', info)
            self.assertIn('available', info)
            self.assertIn('deployed', info)
            self.assertIn('utilization_rate', info)

class TestOrchestratorIntegration(unittest.TestCase):
    """Integration tests for orchestrator components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.pipeline = DisasterResponsePipeline()
        self.resource_allocator = EmergencyResourceAllocator()
    
    @patch('src.vision_service.damage_analyzer.CerebrasVisionAnalyzer.analyze_disaster_image')
    @patch('src.llm_service.text_analyzer.EmergencyTextAnalyzer.analyze_text')
    @patch('src.llm_service.action_generator.EmergencyActionGenerator.generate_action_plan')
    def test_full_orchestrator_pipeline(self, mock_action, mock_text, mock_vision):
        """Test complete orchestrator pipeline"""
        # Mock component responses
        mock_vision.return_value = {
            'disaster_type': 'fire',
            'severity': 8.0,
            'confidence': 0.9
        }
        
        mock_text.return_value = {
            'urgency_level': 'HIGH',
            'key_phrases': ['fire', 'trapped']
        }
        
        mock_action.return_value = {
            'immediate_actions': ['Deploy fire trucks'],
            'resource_requirements': {'firefighters': 8}
        }
        
        # Process complete incident
        incident_data = {
            'text_content': 'Building fire emergency',
            'image_data': b'test_image',
            'location': {'lat': 37.7749, 'lng': -122.4194}
        }
        
        # Step 1: Process incident
        analysis_result = asyncio.run(
            self.pipeline.process_incident(incident_data)
        )
        
        # Step 2: Allocate resources
        allocation_result = asyncio.run(
            self.resource_allocator.allocate_resources(
                analysis_result['incident_analysis'],
                location=incident_data['location']
            )
        )
        
        # Verify end-to-end processing
        self.assertEqual(analysis_result['status'], 'SUCCESS')
        self.assertEqual(allocation_result['allocation_status'], 'SUCCESS')
    
    def test_error_propagation(self):
        """Test error propagation through pipeline"""
        # Test with invalid data that should cause errors
        invalid_data = {'invalid': 'data'}
        
        result = asyncio.run(
            self.pipeline.process_incident(invalid_data)
        )
        
        # Should handle errors gracefully
        self.assertEqual(result['status'], 'ERROR')
        self.assertIn('error', result)
    
    def test_concurrent_processing(self):
        """Test concurrent incident processing"""
        incidents = [
            {
                'text_content': f'Emergency incident {i}',
                'location': {'lat': 37.7749, 'lng': -122.4194}
            }
            for i in range(3)
        ]
        
        # Process incidents concurrently
        async def process_all():
            tasks = [
                self.pipeline.process_incident(incident)
                for incident in incidents
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        results = asyncio.run(process_all())
        
        # Should handle concurrent processing
        self.assertEqual(len(results), 3)
        for result in results:
            if not isinstance(result, Exception):
                self.assertIn('status', result)

def run_orchestrator_tests():
    """Run all orchestrator tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestOrchestrator))
    suite.addTests(loader.loadTestsFromTestCase(TestOrchestratorIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Set up test environment
    os.environ['CEREBRAS_API_KEY'] = 'test_key'
    os.environ['GEMINI_API_KEY'] = 'test_key'
    
    success = run_orchestrator_tests()
    if success:
        print("✅ All orchestrator tests passed!")
    else:
        print("❌ Some orchestrator tests failed!")
        sys.exit(1)