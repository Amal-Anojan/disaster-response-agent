import unittest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import io
import base64

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_service.damage_analyzer import CerebrasVisionAnalyzer
from vision_service.image_preprocessor import ImagePreprocessor
from vision_service.severity_calculator import SeverityCalculator

class TestVisionService(unittest.TestCase):
    """Test cases for vision service components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vision_analyzer = CerebrasVisionAnalyzer()
        self.image_preprocessor = ImagePreprocessor()
        self.severity_calculator = SeverityCalculator()
        
        # Create test image
        self.test_image = Image.new('RGB', (640, 480), color='red')
        self.test_image_bytes = self._image_to_bytes(self.test_image)
    
    def _image_to_bytes(self, image):
        """Convert PIL image to bytes"""
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        return img_buffer.getvalue()
    
    def test_image_preprocessor_resize_image(self):
        """Test image resizing functionality"""
        # Test resizing
        resized_bytes = self.image_preprocessor.resize_image(
            self.test_image_bytes, 
            target_size=(320, 240)
        )
        
        # Verify image was resized
        resized_image = Image.open(io.BytesIO(resized_bytes))
        self.assertEqual(resized_image.size, (320, 240))
    
    def test_image_preprocessor_enhance_image(self):
        """Test image enhancement functionality"""
        enhanced_bytes = self.image_preprocessor.enhance_image(
            self.test_image_bytes,
            brightness=1.2,
            contrast=1.1,
            sharpness=1.0
        )
        
        # Should return enhanced image bytes
        self.assertIsInstance(enhanced_bytes, bytes)
        self.assertTrue(len(enhanced_bytes) > 0)
    
    def test_image_preprocessor_detect_edges(self):
        """Test edge detection functionality"""
        edge_info = self.image_preprocessor.detect_edges(self.test_image_bytes)
        
        # Should return edge information
        self.assertIn('edge_density', edge_info)
        self.assertIn('dominant_edges', edge_info)
        self.assertIsInstance(edge_info['edge_density'], float)
    
    def test_image_preprocessor_analyze_colors(self):
        """Test color analysis functionality"""
        color_analysis = self.image_preprocessor.analyze_colors(self.test_image_bytes)
        
        # Should return color analysis
        self.assertIn('dominant_colors', color_analysis)
        self.assertIn('color_distribution', color_analysis)
        self.assertIsInstance(color_analysis['dominant_colors'], list)
    
    @patch('requests.post')
    def test_vision_analyzer_with_mock_api(self, mock_post):
        """Test vision analyzer with mocked API response"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "This image shows fire damage to a building with severe structural compromise."
                }
            }]
        }
        mock_post.return_value = mock_response
        
        # Test analysis
        result = asyncio.run(
            self.vision_analyzer.analyze_disaster_image(self.test_image_bytes)
        )
        
        # Verify result structure
        self.assertIn('disaster_type', result)
        self.assertIn('severity', result)
        self.assertIn('confidence', result)
        self.assertIn('hazards', result)
    
    @patch('requests.post')
    def test_vision_analyzer_api_error(self, mock_post):
        """Test vision analyzer API error handling"""
        # Mock API error
        mock_post.side_effect = Exception("API Error")
        
        # Test error handling
        result = asyncio.run(
            self.vision_analyzer.analyze_disaster_image(self.test_image_bytes)
        )
        
        # Should return fallback response
        self.assertEqual(result['disaster_type'], 'unknown')
        self.assertEqual(result['confidence'], 0.3)
        self.assertIn('error', result)
    
    def test_severity_calculator_calculate_severity(self):
        """Test severity calculation"""
        test_factors = {
            'structural_damage': 0.8,
            'fire_intensity': 0.7,
            'affected_area': 0.6,
            'casualties_visible': True,
            'evacuation_needed': True
        }
        
        severity = self.severity_calculator.calculate_severity_score(test_factors)
        
        # Should return severity between 1-10
        self.assertIsInstance(severity, float)
        self.assertTrue(1.0 <= severity <= 10.0)
    
    def test_severity_calculator_assess_infrastructure_damage(self):
        """Test infrastructure damage assessment"""
        # Create mock detection results
        mock_detections = [
            {'class': 'building', 'confidence': 0.9, 'damage_level': 'severe'},
            {'class': 'road', 'confidence': 0.8, 'damage_level': 'moderate'},
            {'class': 'vehicle', 'confidence': 0.7, 'damage_level': 'minor'}
        ]
        
        assessment = self.severity_calculator.assess_infrastructure_damage(mock_detections)
        
        # Should return assessment with severity score
        self.assertIn('overall_severity', assessment)
        self.assertIn('damaged_structures', assessment)
        self.assertIn('critical_infrastructure_affected', assessment)
    
    def test_severity_calculator_estimate_affected_population(self):
        """Test population estimation"""
        image_analysis = {
            'detected_objects': {
                'buildings': [{'type': 'residential', 'size': 'large'}],
                'vehicles': [{'type': 'car'}, {'type': 'bus'}],
                'people': [{'visible': True}, {'visible': True}]
            },
            'area_analysis': {
                'urban_density': 'high',
                'building_density': 'medium'
            }
        }
        
        population = self.severity_calculator.estimate_affected_population(image_analysis)
        
        # Should return reasonable population estimate
        self.assertIsInstance(population, int)
        self.assertTrue(population >= 0)
    
    def test_severity_calculator_detect_hazards(self):
        """Test hazard detection"""
        mock_analysis = {
            'fire_detected': True,
            'smoke_density': 0.8,
            'structural_collapse': True,
            'flooding': False,
            'chemical_spill': False
        }
        
        hazards = self.severity_calculator.detect_environmental_hazards(mock_analysis)
        
        # Should return list of detected hazards
        self.assertIsInstance(hazards, list)
        self.assertTrue(len(hazards) > 0)
        self.assertIn('fire', hazards)
    
    def test_severity_calculator_time_criticality(self):
        """Test time criticality assessment"""
        disaster_info = {
            'disaster_type': 'fire',
            'severity': 8.5,
            'fire_spread_rate': 'fast',
            'people_trapped': True,
            'escape_routes_blocked': True
        }
        
        time_criticality = self.severity_calculator.assess_time_criticality(disaster_info)
        
        # Should return time criticality info
        self.assertIn('urgency_level', time_criticality)
        self.assertIn('estimated_safe_time_minutes', time_criticality)
        self.assertIn('critical_factors', time_criticality)
    
    def test_image_validation(self):
        """Test image validation functionality"""
        # Test valid image
        self.assertTrue(
            self.image_preprocessor.validate_image(self.test_image_bytes)
        )
        
        # Test invalid image data
        invalid_data = b"not an image"
        self.assertFalse(
            self.image_preprocessor.validate_image(invalid_data)
        )
    
    def test_batch_processing(self):
        """Test batch image processing"""
        # Create multiple test images
        images = [self.test_image_bytes for _ in range(3)]
        
        # Test batch preprocessing
        results = self.image_preprocessor.batch_preprocess(images)
        
        # Should return results for all images
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('processed', result)
            self.assertIn('metadata', result)
    
    def tearDown(self):
        """Clean up after tests"""
        pass

class TestVisionServiceIntegration(unittest.TestCase):
    """Integration tests for vision service components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.vision_analyzer = CerebrasVisionAnalyzer()
        self.image_preprocessor = ImagePreprocessor()
        self.severity_calculator = SeverityCalculator()
        
        # Create test disaster image (red for fire simulation)
        self.fire_image = Image.new('RGB', (640, 480), color=(255, 100, 0))
        self.fire_image_bytes = self._image_to_bytes(self.fire_image)
    
    def _image_to_bytes(self, image):
        """Convert PIL image to bytes"""
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        return img_buffer.getvalue()
    
    @patch('requests.post')
    def test_full_vision_pipeline(self, mock_post):
        """Test complete vision processing pipeline"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Severe fire damage with structural collapse, multiple casualties visible, immediate evacuation required."
                }
            }]
        }
        mock_post.return_value = mock_response
        
        # Step 1: Preprocess image
        enhanced_image = self.image_preprocessor.enhance_image(
            self.fire_image_bytes,
            brightness=1.1,
            contrast=1.2
        )
        
        # Step 2: Analyze with vision AI
        analysis = asyncio.run(
            self.vision_analyzer.analyze_disaster_image(enhanced_image)
        )
        
        # Step 3: Calculate severity
        severity_factors = {
            'structural_damage': analysis.get('damage_assessment', {}).get('structural', 0.5),
            'fire_intensity': 0.8,
            'affected_area': 0.7,
            'casualties_visible': True,
            'evacuation_needed': True
        }
        
        severity_score = self.severity_calculator.calculate_severity_score(severity_factors)
        
        # Verify pipeline results
        self.assertIsInstance(enhanced_image, bytes)
        self.assertIn('disaster_type', analysis)
        self.assertIsInstance(severity_score, float)
        self.assertTrue(1.0 <= severity_score <= 10.0)
    
    def test_error_resilience(self):
        """Test system resilience to errors"""
        # Test with invalid image data
        invalid_data = b"invalid image data"
        
        # Should handle gracefully without crashing
        try:
            self.image_preprocessor.enhance_image(invalid_data)
        except Exception as e:
            # Should be a handled exception, not a crash
            self.assertIsInstance(e, (ValueError, IOError))

def run_vision_service_tests():
    """Run all vision service tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestVisionService))
    suite.addTests(loader.loadTestsFromTestCase(TestVisionServiceIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Set up test environment
    os.environ['CEREBRAS_API_KEY'] = 'test_key'
    
    success = run_vision_service_tests()
    if success:
        print("✅ All vision service tests passed!")
    else:
        print("❌ Some vision service tests failed!")
        sys.exit(1)