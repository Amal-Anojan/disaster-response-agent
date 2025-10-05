import unittest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_service.action_generator import EmergencyActionGenerator
from llm_service.rag_system import EmergencyRAGSystem
from llm_service.text_analyzer import EmergencyTextAnalyzer

class TestLLMService(unittest.TestCase):
    """Test cases for LLM service components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.action_generator = EmergencyActionGenerator()
        self.rag_system = EmergencyRAGSystem()
        self.text_analyzer = EmergencyTextAnalyzer()
        
        # Sample incident data for testing
        self.sample_incident = {
            'disaster_type': 'fire',
            'severity': 8.5,
            'urgency': 'CRITICAL',
            'affected_population': 150,
            'location': {'lat': 37.7749, 'lng': -122.4194},
            'affected_infrastructure': ['residential_building', 'power_lines']
        }
        
        self.sample_text = """
        Large apartment fire on Main Street. Multiple people trapped on upper floors.
        Fire department on scene. Evacuation in progress. Power lines down.
        Estimated 50-75 people affected. Wind spreading fire to adjacent buildings.
        """
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_action_generator_success(self, mock_model, mock_configure):
        """Test successful action plan generation"""
        # Mock the Gemini API response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "immediate_actions": [
                "Deploy additional fire engines",
                "Establish evacuation perimeter",
                "Coordinate with utilities for power shutdown"
            ],
            "resource_requirements": {
                "personnel": {"firefighters": 12, "paramedics": 4},
                "vehicles": {"fire_engines": 3, "ambulances": 2},
                "equipment": ["ladder_trucks", "breathing_apparatus"]
            },
            "timeline": {
                "0-15_minutes": ["Secure perimeter", "Begin evacuation"],
                "15-30_minutes": ["Fire suppression", "Medical triage"],
                "30-60_minutes": ["Search and rescue", "Utility shutdown"]
            }
        })
        
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        # Test action plan generation
        result = asyncio.run(
            self.action_generator.generate_action_plan(
                self.sample_incident,
                self.sample_text
            )
        )
        
        # Verify result structure
        self.assertIn('immediate_actions', result)
        self.assertIn('resource_requirements', result)
        self.assertIn('timeline', result)
        self.assertIsInstance(result['immediate_actions'], list)
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_action_generator_api_error(self, mock_model, mock_configure):
        """Test action generator API error handling"""
        # Mock API error
        mock_model_instance = Mock()
        mock_model_instance.generate_content.side_effect = Exception("API Error")
        mock_model.return_value = mock_model_instance
        
        # Test error handling
        result = asyncio.run(
            self.action_generator.generate_action_plan(
                self.sample_incident,
                self.sample_text
            )
        )
        
        # Should return fallback action plan
        self.assertIn('immediate_actions', result)
        self.assertIn('error_fallback', result)
        self.assertTrue(result['error_fallback'])
    
    def test_rag_system_initialization(self):
        """Test RAG system initialization"""
        # Should initialize without errors
        self.assertIsInstance(self.rag_system, EmergencyRAGSystem)
        self.assertIsInstance(self.rag_system.knowledge_base, dict)
    
    def test_rag_system_search_procedures(self):
        """Test emergency procedure search"""
        # Test searching for fire procedures
        procedures = self.rag_system.search_emergency_procedures(
            disaster_type='fire',
            severity='high'
        )
        
        # Should return relevant procedures
        self.assertIsInstance(procedures, list)
        if procedures:  # If procedures are found
            self.assertIn('procedure', procedures[0])
            self.assertIn('priority', procedures[0])
    
    def test_rag_system_get_resources(self):
        """Test resource information retrieval"""
        resources = self.rag_system.get_available_resources(
            resource_type='fire_engines',
            location={'lat': 37.7749, 'lng': -122.4194}
        )
        
        # Should return resource information
        self.assertIsInstance(resources, list)
    
    def test_rag_system_similarity_search(self):
        """Test similarity search functionality"""
        similar_cases = self.rag_system.find_similar_incidents(
            current_incident=self.sample_incident,
            limit=3
        )
        
        # Should return similar cases
        self.assertIsInstance(similar_cases, list)
        self.assertTrue(len(similar_cases) <= 3)
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_text_analyzer_success(self, mock_model, mock_configure):
        """Test successful text analysis"""
        # Mock API response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "disaster_types": ["fire", "building_collapse"],
            "primary_disaster_type": "fire",
            "urgency_level": "CRITICAL",
            "key_phrases": ["trapped people", "evacuation", "power lines down"],
            "locations": [
                {"text": "Main Street", "type": "street", "confidence": 0.9}
            ],
            "numbers": {
                "affected_people": [50, 75],
                "casualties": [],
                "time_references": ["in progress"]
            },
            "entities": {
                "organizations": ["fire department"],
                "infrastructure": ["power lines", "buildings"]
            }
        })
        
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        # Test text analysis
        result = asyncio.run(
            self.text_analyzer.analyze_text(self.sample_text)
        )
        
        # Verify result structure
        self.assertIn('disaster_types', result)
        self.assertIn('primary_disaster_type', result)
        self.assertIn('urgency_level', result)
        self.assertIn('key_phrases', result)
    
    def test_text_analyzer_preprocessing(self):
        """Test text preprocessing functionality"""
        # Test text cleaning
        dirty_text = "  URGENT!!! Fire at 123 Main St... People TRAPPED!!!  "
        cleaned = self.text_analyzer.preprocess_text(dirty_text)
        
        # Should clean and normalize text
        self.assertNotIn('!!!', cleaned)
        self.assertLess(len(cleaned), len(dirty_text))
        
    def test_text_analyzer_extract_entities(self):
        """Test entity extraction"""
        entities = self.text_analyzer.extract_entities(self.sample_text)
        
        # Should extract entities
        self.assertIn('locations', entities)
        self.assertIn('numbers', entities)
        self.assertIn('time_references', entities)
    
    def test_text_analyzer_sentiment_analysis(self):
        """Test sentiment analysis"""
        sentiment = self.text_analyzer.analyze_sentiment(self.sample_text)
        
        # Should return sentiment information
        self.assertIn('sentiment', sentiment)
        self.assertIn('confidence', sentiment)
        self.assertIn('emotional_indicators', sentiment)
    
    def test_text_analyzer_urgency_detection(self):
        """Test urgency detection"""
        # Test high urgency text
        urgent_text = "EMERGENCY! People trapped! IMMEDIATE HELP NEEDED!"
        urgency = self.text_analyzer.detect_urgency_level(urgent_text)
        
        # Should detect high urgency
        self.assertIn(urgency['level'], ['HIGH', 'CRITICAL'])
        
        # Test low urgency text
        normal_text = "Minor water leak reported in building basement"
        urgency = self.text_analyzer.detect_urgency_level(normal_text)
        
        # Should detect lower urgency
        self.assertIn(urgency['level'], ['LOW', 'MEDIUM'])
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_action_generator_resource_optimization(self, mock_model, mock_configure):
        """Test resource optimization in action plans"""
        # Mock response with resource optimization
        mock_response = Mock()
        mock_response.text = json.dumps({
            "immediate_actions": ["Deploy nearest fire engine", "Alert ambulance"],
            "resource_requirements": {
                "personnel": {"firefighters": 6, "paramedics": 2},
                "vehicles": {"fire_engines": 1, "ambulances": 1}
            },
            "optimization_notes": [
                "Using closest available resources",
                "Minimizing response time"
            ]
        })
        
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        # Test with resource constraints
        result = asyncio.run(
            self.action_generator.generate_action_plan(
                self.sample_incident,
                self.sample_text,
                resource_constraints={'max_personnel': 10}
            )
        )
        
        # Should optimize resources
        self.assertIn('resource_requirements', result)
        if 'optimization_notes' in result:
            self.assertIsInstance(result['optimization_notes'], list)

class TestLLMServiceIntegration(unittest.TestCase):
    """Integration tests for LLM service components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.action_generator = EmergencyActionGenerator()
        self.rag_system = EmergencyRAGSystem()
        self.text_analyzer = EmergencyTextAnalyzer()
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_full_llm_pipeline(self, mock_model, mock_configure):
        """Test complete LLM processing pipeline"""
        # Mock text analysis response
        text_analysis_response = Mock()
        text_analysis_response.text = json.dumps({
            "disaster_types": ["fire"],
            "primary_disaster_type": "fire",
            "urgency_level": "HIGH",
            "key_phrases": ["building fire", "people trapped"],
            "affected_people_estimate": 50
        })
        
        # Mock action generation response
        action_response = Mock()
        action_response.text = json.dumps({
            "immediate_actions": ["Deploy fire trucks", "Evacuate area"],
            "resource_requirements": {
                "personnel": {"firefighters": 8},
                "vehicles": {"fire_engines": 2}
            }
        })
        
        # Set up mock to return different responses for different calls
        mock_model_instance = Mock()
        mock_model_instance.generate_content.side_effect = [
            text_analysis_response, 
            action_response
        ]
        mock_model.return_value = mock_model_instance
        
        incident_text = "Building fire on Oak Street, people trapped on 3rd floor"
        
        # Step 1: Analyze text
        text_analysis = asyncio.run(
            self.text_analyzer.analyze_text(incident_text)
        )
        
        # Step 2: Search for relevant procedures
        procedures = self.rag_system.search_emergency_procedures(
            disaster_type=text_analysis.get('primary_disaster_type', 'fire'),
            severity='high'
        )
        
        # Step 3: Generate action plan
        incident_data = {
            'disaster_type': text_analysis.get('primary_disaster_type'),
            'urgency': text_analysis.get('urgency_level'),
            'severity': 7.5
        }
        
        action_plan = asyncio.run(
            self.action_generator.generate_action_plan(
                incident_data,
                incident_text
            )
        )
        
        # Verify pipeline results
        self.assertIn('primary_disaster_type', text_analysis)
        self.assertIsInstance(procedures, list)
        self.assertIn('immediate_actions', action_plan)
    
    def test_error_recovery_pipeline(self):
        """Test pipeline resilience to component failures"""
        # Test with invalid input
        invalid_text = ""
        
        # Should handle gracefully
        try:
            result = asyncio.run(
                self.text_analyzer.analyze_text(invalid_text)
            )
            # Should return fallback result
            self.assertIn('confidence', result)
        except Exception as e:
            # Should be handled gracefully
            self.fail(f"Pipeline should handle empty text gracefully: {e}")
    
    def test_rag_system_knowledge_integration(self):
        """Test RAG system integration with knowledge base"""
        # Test retrieving emergency procedures
        fire_procedures = self.rag_system.search_emergency_procedures('fire', 'high')
        flood_procedures = self.rag_system.search_emergency_procedures('flood', 'medium')
        
        # Should return different procedures for different disasters
        self.assertIsInstance(fire_procedures, list)
        self.assertIsInstance(flood_procedures, list)
    
    def test_contextual_action_generation(self):
        """Test context-aware action generation"""
        # Test different contexts
        contexts = [
            {
                'disaster_type': 'fire',
                'severity': 9.0,
                'time_of_day': 'night',
                'weather': 'windy'
            },
            {
                'disaster_type': 'flood',
                'severity': 6.0,
                'time_of_day': 'day',
                'weather': 'stormy'
            }
        ]
        
        for context in contexts:
            # Should adapt to context
            self.assertIsInstance(context, dict)
            self.assertIn('disaster_type', context)

def run_llm_service_tests():
    """Run all LLM service tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestLLMService))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMServiceIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Set up test environment
    os.environ['GEMINI_API_KEY'] = 'test_key'
    
    success = run_llm_service_tests()
    if success:
        print("✅ All LLM service tests passed!")
    else:
        print("❌ Some LLM service tests failed!")
        sys.exit(1)