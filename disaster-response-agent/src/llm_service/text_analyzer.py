import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class EmergencyTextAnalyzer:
    def __init__(self):
        """Initialize text analyzer for emergency content"""
        
        # Emergency keywords by category
        self.emergency_keywords = {
            'fire': ['fire', 'burning', 'smoke', 'flames', 'ignite', 'blaze', 'combustion', 'arson'],
            'flood': ['flood', 'flooding', 'water', 'overflow', 'dam breach', 'levee', 'tsunami', 'storm surge'],
            'earthquake': ['earthquake', 'quake', 'tremor', 'seismic', 'shake', 'fault', 'aftershock'],
            'medical': ['injury', 'injured', 'casualty', 'hurt', 'bleeding', 'unconscious', 'medical', 'ambulance'],
            'storm': ['storm', 'tornado', 'hurricane', 'wind', 'hail', 'lightning', 'cyclone'],
            'explosion': ['explosion', 'blast', 'bomb', 'detonation', 'explode', 'gas leak'],
            'accident': ['accident', 'crash', 'collision', 'vehicle', 'traffic', 'derailment'],
            'hazmat': ['chemical', 'toxic', 'hazardous', 'spill', 'contamination', 'gas', 'radiation']
        }
        
        # Urgency indicators
        self.urgency_keywords = {
            'critical': ['critical', 'urgent', 'immediate', 'emergency', 'help', 'trapped', 'dying', 'fatal'],
            'high': ['serious', 'severe', 'major', 'spreading', 'multiple', 'casualties', 'evacuation'],
            'medium': ['moderate', 'caution', 'warning', 'concern', 'developing', 'potential'],
            'low': ['minor', 'small', 'isolated', 'contained', 'stable', 'monitoring']
        }
        
        # Location extraction patterns
        self.location_patterns = [
            r'at\s+([A-Z][A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln))',
            r'on\s+([A-Z][A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln))',
            r'near\s+([A-Z][A-Za-z\s]+)',
            r'in\s+([A-Z][A-Za-z\s]+)',
            r'([A-Z][A-Za-z\s]+ (?:Hospital|School|Mall|Center|Building|Bridge))'
        ]
        
        # Number extraction for casualties, affected people
        self.number_patterns = [
            r'(\d+)\s+(?:people|persons|individuals|casualties|injured|trapped|affected)',
            r'(\d+)\s+(?:buildings|homes|houses|vehicles|cars)',
            r'approximately\s+(\d+)',
            r'about\s+(\d+)',
            r'over\s+(\d+)',
            r'more than\s+(\d+)'
        ]
    
    async def analyze_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze emergency text for key information
        
        Args:
            text: Text content to analyze
            metadata: Additional metadata about the text
            
        Returns:
            Analysis results with disaster type, urgency, locations, etc.
        """
        try:
            if not text or not text.strip():
                return self._create_empty_analysis()
            
            # Clean and normalize text
            cleaned_text = self._clean_text(text)
            
            # Extract disaster type
            disaster_types = self._extract_disaster_types(cleaned_text)
            
            # Determine urgency level
            urgency_level = self._determine_urgency(cleaned_text)
            
            # Extract locations
            locations = self._extract_locations(cleaned_text)
            
            # Extract numbers (casualties, affected people)
            numbers = self._extract_numbers(cleaned_text)
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(cleaned_text)
            
            # Sentiment analysis (basic)
            sentiment = self._analyze_sentiment(cleaned_text)
            
            # Extract temporal information
            temporal_info = self._extract_temporal_info(cleaned_text)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(cleaned_text, disaster_types, urgency_level)
            
            # Extract specific emergency details
            emergency_details = self._extract_emergency_details(cleaned_text, disaster_types)
            
            return {
                'disaster_types': disaster_types,
                'primary_disaster_type': disaster_types[0] if disaster_types else 'unknown',
                'urgency_level': urgency_level,
                'locations': locations,
                'numbers': numbers,
                'key_phrases': key_phrases,
                'sentiment': sentiment,
                'temporal_info': temporal_info,
                'confidence': confidence,
                'emergency_details': emergency_details,
                'original_text': text,
                'cleaned_text': cleaned_text,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._create_error_analysis(text, str(e))
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation
        cleaned = re.sub(r'[^\w\s.,!?;:()\-]', '', cleaned)
        
        # Convert to lowercase for analysis (but keep original case in result)
        return cleaned
    
    def _extract_disaster_types(self, text: str) -> List[str]:
        """Extract disaster types from text"""
        text_lower = text.lower()
        found_types = []
        
        for disaster_type, keywords in self.emergency_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if disaster_type not in found_types:
                        found_types.append(disaster_type)
                    break
        
        # Sort by relevance (number of matching keywords)
        type_scores = {}
        for disaster_type in found_types:
            score = sum(1 for keyword in self.emergency_keywords[disaster_type] if keyword in text_lower)
            type_scores[disaster_type] = score
        
        # Return sorted by score (most relevant first)
        return sorted(found_types, key=lambda x: type_scores.get(x, 0), reverse=True)
    
    def _determine_urgency(self, text: str) -> str:
        """Determine urgency level from text"""
        text_lower = text.lower()
        
        # Count keywords for each urgency level
        urgency_scores = {}
        for level, keywords in self.urgency_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            urgency_scores[level] = score
        
        # Additional urgency indicators
        if any(word in text_lower for word in ['!!!', 'asap', 'now', 'immediately', 'urgent']):
            urgency_scores['critical'] += 2
        
        if any(word in text_lower for word in ['please', 'help', 'emergency']):
            urgency_scores['high'] += 1
        
        # Determine highest scoring urgency level
        if urgency_scores['critical'] > 0:
            return 'CRITICAL'
        elif urgency_scores['high'] > 0:
            return 'HIGH'
        elif urgency_scores['medium'] > 0:
            return 'MEDIUM'
        elif urgency_scores['low'] > 0:
            return 'LOW'
        else:
            return 'MEDIUM'  # Default
    
    def _extract_locations(self, text: str) -> List[Dict[str, Any]]:
        """Extract location information from text"""
        locations = []
        
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                
                location = {
                    'text': match.strip(),
                    'type': self._classify_location_type(match),
                    'confidence': 0.8 if 'street' in match.lower() or 'avenue' in match.lower() else 0.6
                }
                
                # Avoid duplicates
                if not any(loc['text'].lower() == location['text'].lower() for loc in locations):
                    locations.append(location)
        
        return locations[:5]  # Return top 5 locations
    
    def _classify_location_type(self, location: str) -> str:
        """Classify the type of location"""
        location_lower = location.lower()
        
        if any(word in location_lower for word in ['street', 'avenue', 'road', 'boulevard', 'drive', 'lane']):
            return 'address'
        elif any(word in location_lower for word in ['hospital', 'medical', 'clinic']):
            return 'medical_facility'
        elif any(word in location_lower for word in ['school', 'university', 'college']):
            return 'educational'
        elif any(word in location_lower for word in ['mall', 'center', 'plaza', 'building']):
            return 'commercial'
        elif any(word in location_lower for word in ['bridge', 'tunnel', 'highway', 'freeway']):
            return 'infrastructure'
        elif any(word in location_lower for word in ['park', 'beach', 'lake', 'river']):
            return 'natural'
        else:
            return 'general'
    
    def _extract_numbers(self, text: str) -> Dict[str, List[int]]:
        """Extract numerical information from text"""
        numbers = {
            'casualties': [],
            'affected_people': [],
            'structures': [],
            'general': []
        }
        
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    num = int(match)
                    
                    # Categorize the number based on context
                    context = text[max(0, text.find(match) - 50):text.find(match) + 50].lower()
                    
                    if any(word in context for word in ['injured', 'casualty', 'hurt', 'dead', 'killed']):
                        numbers['casualties'].append(num)
                    elif any(word in context for word in ['people', 'persons', 'affected', 'evacuated']):
                        numbers['affected_people'].append(num)
                    elif any(word in context for word in ['building', 'house', 'structure', 'vehicle']):
                        numbers['structures'].append(num)
                    else:
                        numbers['general'].append(num)
                        
                except ValueError:
                    continue
        
        return numbers
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from emergency text"""
        # Simple key phrase extraction based on emergency context
        key_phrases = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Ignore very short sentences
                
                # Check if sentence contains emergency keywords
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keywords in self.emergency_keywords.values() for keyword in keywords):
                    # Extract meaningful phrases (noun phrases, action phrases)
                    phrases = self._extract_noun_phrases(sentence)
                    key_phrases.extend(phrases)
        
        # Remove duplicates and return top phrases
        unique_phrases = []
        for phrase in key_phrases:
            if phrase not in unique_phrases and len(phrase) > 5:
                unique_phrases.append(phrase)
        
        return unique_phrases[:10]  # Return top 10 key phrases
    
    def _extract_noun_phrases(self, sentence: str) -> List[str]:
        """Extract noun phrases from sentence (simplified)"""
        # Simple noun phrase extraction based on patterns
        patterns = [
            r'(?:the|a|an)?\s*(?:\w+\s+)*(?:fire|building|vehicle|person|area|structure|road|bridge)',
            r'\b(?:\w+\s+){0,2}(?:emergency|disaster|incident|accident|explosion)\b',
            r'\b(?:major|severe|critical|massive|large)\s+\w+',
        ]
        
        phrases = []
        for pattern in patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            phrases.extend([match.strip() for match in matches])
        
        return phrases
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis for emergency text"""
        text_lower = text.lower()
        
        # Positive/negative word counts
        positive_words = ['safe', 'secure', 'contained', 'stable', 'rescued', 'recovered', 'successful']
        negative_words = ['danger', 'critical', 'severe', 'trapped', 'destroyed', 'dead', 'injured', 'panic']
        urgent_words = ['urgent', 'immediate', 'emergency', 'critical', 'help', 'asap']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        urgent_count = sum(1 for word in urgent_words if word in text_lower)
        
        # Calculate sentiment scores
        total_words = len(text_lower.split())
        positive_ratio = positive_count / max(total_words, 1)
        negative_ratio = negative_count / max(total_words, 1)
        urgent_ratio = urgent_count / max(total_words, 1)
        
        # Determine overall sentiment
        if urgent_ratio > 0.02:  # 2% of words are urgent
            sentiment_label = 'urgent'
        elif negative_ratio > positive_ratio * 1.5:
            sentiment_label = 'negative'
        elif positive_ratio > negative_ratio * 1.5:
            sentiment_label = 'positive'
        else:
            sentiment_label = 'neutral'
        
        return {
            'label': sentiment_label,
            'positive_score': positive_ratio,
            'negative_score': negative_ratio,
            'urgent_score': urgent_ratio,
            'confidence': min(max(abs(positive_ratio - negative_ratio), urgent_ratio), 1.0)
        }
    
    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Extract temporal information from text"""
        temporal_patterns = [
            r'(\d{1,2}:\d{2}(?:\s*(?:AM|PM|am|pm))?)',  # Times
            r'(now|currently|ongoing|just happened|minutes ago|hours ago)',  # Relative time
            r'(today|yesterday|tonight|this morning|this afternoon|this evening)',  # Relative dates
            r'(\d{1,2}/\d{1,2}/\d{2,4})',  # Dates
        ]
        
        temporal_info = {
            'times': [],
            'relative_times': [],
            'dates': [],
            'urgency_indicators': []
        }
        
        text_lower = text.lower()
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if ':' in match:  # Time format
                    temporal_info['times'].append(match)
                elif any(word in match.lower() for word in ['now', 'currently', 'just', 'ago']):
                    temporal_info['relative_times'].append(match)
                    if any(word in match.lower() for word in ['now', 'just']):
                        temporal_info['urgency_indicators'].append(match)
                elif '/' in match:  # Date format
                    temporal_info['dates'].append(match)
                else:
                    temporal_info['relative_times'].append(match)
        
        return temporal_info
    
    def _calculate_confidence(self, text: str, disaster_types: List[str], urgency_level: str) -> float:
        """Calculate confidence score for the analysis"""
        confidence_factors = []
        
        # Text length factor
        text_length = len(text.split())
        if text_length > 50:
            confidence_factors.append(0.9)
        elif text_length > 20:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Disaster type detection confidence
        if disaster_types:
            # More disaster types detected = higher confidence if they're related
            if len(disaster_types) == 1:
                confidence_factors.append(0.8)
            elif len(disaster_types) <= 3:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)  # Too many types might indicate confusion
        else:
            confidence_factors.append(0.3)
        
        # Urgency detection confidence
        urgency_confidence_map = {
            'CRITICAL': 0.9,
            'HIGH': 0.8,
            'MEDIUM': 0.6,
            'LOW': 0.7
        }
        confidence_factors.append(urgency_confidence_map.get(urgency_level, 0.5))
        
        # Emergency keyword density
        text_lower = text.lower()
        emergency_word_count = sum(1 for keywords in self.emergency_keywords.values() for keyword in keywords if keyword in text_lower)
        word_density = emergency_word_count / max(len(text.split()), 1)
        
        if word_density > 0.05:  # 5% emergency words
            confidence_factors.append(0.9)
        elif word_density > 0.02:  # 2% emergency words
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Calculate average confidence
        return sum(confidence_factors) / len(confidence_factors)
    
    def _extract_emergency_details(self, text: str, disaster_types: List[str]) -> Dict[str, Any]:
        """Extract specific emergency details based on disaster type"""
        details = {}
        text_lower = text.lower()
        
        for disaster_type in disaster_types:
            if disaster_type == 'fire':
                details['fire_details'] = {
                    'structure_type': self._extract_structure_type(text),
                    'fire_size': self._extract_fire_size(text),
                    'smoke_visibility': 'smoke' in text_lower,
                    'evacuation_needed': any(word in text_lower for word in ['evacuate', 'evacuation', 'get out'])
                }
            elif disaster_type == 'flood':
                details['flood_details'] = {
                    'water_depth': self._extract_water_depth(text),
                    'water_source': self._extract_water_source(text),
                    'rising_water': any(word in text_lower for word in ['rising', 'increasing', 'getting higher']),
                    'swift_water': any(word in text_lower for word in ['swift', 'fast', 'rapid', 'current'])
                }
            elif disaster_type == 'accident':
                details['accident_details'] = {
                    'vehicle_types': self._extract_vehicle_types(text),
                    'injuries_reported': any(word in text_lower for word in ['injury', 'injured', 'hurt']),
                    'road_blocked': any(word in text_lower for word in ['blocked', 'closed', 'impassable'])
                }
        
        return details
    
    def _extract_structure_type(self, text: str) -> Optional[str]:
        """Extract structure type for fire incidents"""
        structure_types = ['house', 'building', 'apartment', 'office', 'warehouse', 'factory', 'store', 'restaurant']
        text_lower = text.lower()
        
        for structure in structure_types:
            if structure in text_lower:
                return structure
        return None
    
    def _extract_fire_size(self, text: str) -> Optional[str]:
        """Extract fire size indicators"""
        size_indicators = {
            'small': ['small', 'minor', 'contained'],
            'medium': ['medium', 'moderate', 'growing'],
            'large': ['large', 'major', 'massive', 'huge', 'spreading']
        }
        
        text_lower = text.lower()
        for size, keywords in size_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                return size
        return None
    
    def _extract_water_depth(self, text: str) -> Optional[str]:
        """Extract water depth from flood text"""
        depth_patterns = [
            r'(\d+)\s*(?:feet|ft|foot)\s*(?:deep|of water)',
            r'(\d+)\s*(?:inches|in)\s*(?:deep|of water)',
            r'(ankle|knee|waist|chest)\s*deep'
        ]
        
        for pattern in depth_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        return None
    
    def _extract_water_source(self, text: str) -> Optional[str]:
        """Extract water source for flood"""
        sources = ['river', 'creek', 'dam', 'levee', 'storm drain', 'pipe', 'main', 'rain']
        text_lower = text.lower()
        
        for source in sources:
            if source in text_lower:
                return source
        return None
    
    def _extract_vehicle_types(self, text: str) -> List[str]:
        """Extract vehicle types from accident text"""
        vehicles = ['car', 'truck', 'motorcycle', 'bus', 'van', 'semi', 'trailer', 'bicycle']
        text_lower = text.lower()
        
        found_vehicles = []
        for vehicle in vehicles:
            if vehicle in text_lower:
                found_vehicles.append(vehicle)
        
        return found_vehicles
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis result"""
        return {
            'disaster_types': [],
            'primary_disaster_type': 'unknown',
            'urgency_level': 'LOW',
            'locations': [],
            'numbers': {'casualties': [], 'affected_people': [], 'structures': [], 'general': []},
            'key_phrases': [],
            'sentiment': {'label': 'neutral', 'positive_score': 0, 'negative_score': 0, 'urgent_score': 0, 'confidence': 0},
            'temporal_info': {'times': [], 'relative_times': [], 'dates': [], 'urgency_indicators': []},
            'confidence': 0.0,
            'emergency_details': {},
            'original_text': '',
            'cleaned_text': '',
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _create_error_analysis(self, text: str, error_msg: str) -> Dict[str, Any]:
        """Create error analysis result"""
        result = self._create_empty_analysis()
        result.update({
            'original_text': text,
            'error': error_msg,
            'confidence': 0.0
        })
        return result

# Usage example and testing
async def test_text_analyzer():
    """Test the text analyzer"""
    analyzer = EmergencyTextAnalyzer()
    
    # Test texts
    test_texts = [
        "Major apartment fire on 123 Main Street, multiple people trapped on upper floors, immediate evacuation needed",
        "Severe flooding on Highway 101, approximately 50 vehicles stranded, water rising rapidly",
        "Building collapse at Downtown Mall, search and rescue teams needed urgently, casualties reported",
        "Gas leak explosion at industrial plant, hazmat teams required, evacuation of surrounding area"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Test {i+1} ---")
        print(f"Text: {text}")
        
        result = await analyzer.analyze_text(text)
        
        print(f"Disaster Types: {result['disaster_types']}")
        print(f"Urgency: {result['urgency_level']}")
        print(f"Locations: {[loc['text'] for loc in result['locations']]}")
        print(f"Key Phrases: {result['key_phrases'][:3]}")
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_text_analyzer())