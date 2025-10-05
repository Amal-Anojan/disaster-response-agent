import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import chromadb
from chromadb.config import Settings
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class EmergencyRAGSystem:
    def __init__(self, 
                 knowledge_base_path: str = "data/knowledge_base/",
                 collection_name: str = "emergency_procedures"):
        """
        Initialize RAG system for emergency response knowledge
        
        Args:
            knowledge_base_path: Path to knowledge base files
            collection_name: ChromaDB collection name
        """
        self.knowledge_base_path = knowledge_base_path
        self.collection_name = collection_name
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("SentenceTransformer model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        self._initialize_chromadb()
        
        # Load and index knowledge base
        self.knowledge_base = self._load_knowledge_base()
        self._index_knowledge_base()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path="data/chroma_db",
                settings=Settings(allow_reset=True)
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Retrieved existing ChromaDB collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Emergency response procedures and knowledge"}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base from files"""
        knowledge_base = {}
        
        # Knowledge base files to load
        files_to_load = {
            'procedures': 'emergency_procedures.json',
            'teams': 'response_teams.json',
            'resources': 'resources_database.json'
        }
        
        for key, filename in files_to_load.items():
            file_path = os.path.join(self.knowledge_base_path, filename)
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        knowledge_base[key] = json.load(f)
                    logger.info(f"Loaded {key} from {filename}")
                else:
                    logger.warning(f"Knowledge base file not found: {filename}")
                    knowledge_base[key] = self._get_fallback_knowledge(key)
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                knowledge_base[key] = self._get_fallback_knowledge(key)
        
        return knowledge_base
    
    def _get_fallback_knowledge(self, key: str) -> Dict[str, Any]:
        """Get fallback knowledge when files are not available"""
        fallback_knowledge = {
            'procedures': {
                'fire': [
                    "Deploy fire suppression units immediately",
                    "Establish evacuation perimeter",
                    "Set up incident command post",
                    "Coordinate with utility companies"
                ],
                'flood': [
                    "Evacuate low-lying areas",
                    "Deploy water rescue teams",
                    "Set up emergency shelters",
                    "Monitor water levels"
                ],
                'earthquake': [
                    "Deploy search and rescue teams",
                    "Assess structural damage",
                    "Set up triage areas",
                    "Inspect critical infrastructure"
                ]
            },
            'teams': {
                'fire_department': {
                    'specializations': ['fire_suppression', 'hazmat', 'rescue'],
                    'response_time': 8,
                    'capacity': 50
                },
                'medical_services': {
                    'specializations': ['triage', 'emergency_medicine', 'trauma'],
                    'response_time': 10,
                    'capacity': 30
                },
                'police': {
                    'specializations': ['crowd_control', 'traffic_management', 'security'],
                    'response_time': 6,
                    'capacity': 40
                }
            },
            'resources': {
                'vehicles': ['fire_engines', 'ambulances', 'police_cars', 'rescue_vehicles'],
                'equipment': ['medical_supplies', 'rescue_tools', 'communication_devices'],
                'facilities': ['hospitals', 'emergency_shelters', 'command_centers']
            }
        }
        
        return fallback_knowledge.get(key, {})
    
    def _index_knowledge_base(self):
        """Index knowledge base into ChromaDB"""
        if not self.collection or not self.embedding_model:
            logger.warning("Cannot index knowledge base - ChromaDB or embedding model not available")
            return
        
        try:
            # Check if collection already has documents
            existing_count = self.collection.count()
            if existing_count > 0:
                logger.info(f"Knowledge base already indexed ({existing_count} documents)")
                return
            
            documents = []
            metadatas = []
            ids = []
            
            # Index emergency procedures
            if 'procedures' in self.knowledge_base:
                procedures = self.knowledge_base['procedures']
                for disaster_type, procedure_list in procedures.items():
                    for i, procedure in enumerate(procedure_list):
                        if isinstance(procedure, str):
                            documents.append(procedure)
                            metadatas.append({
                                'type': 'procedure',
                                'disaster_type': disaster_type,
                                'procedure_id': i,
                                'content_type': 'emergency_procedure'
                            })
                            ids.append(f"procedure_{disaster_type}_{i}")
            
            # Index team information
            if 'teams' in self.knowledge_base:
                teams = self.knowledge_base['teams']
                for team_name, team_info in teams.items():
                    # Create document from team information
                    team_doc = f"Team: {team_name}. "
                    if 'specializations' in team_info:
                        team_doc += f"Specializations: {', '.join(team_info['specializations'])}. "
                    if 'response_time' in team_info:
                        team_doc += f"Response time: {team_info['response_time']} minutes. "
                    if 'capacity' in team_info:
                        team_doc += f"Capacity: {team_info['capacity']} personnel."
                    
                    documents.append(team_doc)
                    metadatas.append({
                        'type': 'team',
                        'team_name': team_name,
                        'content_type': 'team_information'
                    })
                    ids.append(f"team_{team_name}")
            
            # Index resources
            if 'resources' in self.knowledge_base:
                resources = self.knowledge_base['resources']
                for resource_type, resource_list in resources.items():
                    if isinstance(resource_list, list):
                        resource_doc = f"Available {resource_type}: {', '.join(resource_list)}"
                        documents.append(resource_doc)
                        metadatas.append({
                            'type': 'resource',
                            'resource_type': resource_type,
                            'content_type': 'resource_information'
                        })
                        ids.append(f"resource_{resource_type}")
            
            # Add documents to ChromaDB
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Indexed {len(documents)} documents into ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to index knowledge base: {e}")
    
    def retrieve_relevant_context(self, 
                                query: str, 
                                disaster_type: Optional[str] = None,
                                content_types: Optional[List[str]] = None,
                                max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for emergency response
        
        Args:
            query: Query text to find relevant information
            disaster_type: Specific disaster type to filter by
            content_types: List of content types to include
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant context documents
        """
        try:
            if not self.collection:
                logger.warning("ChromaDB collection not available, using fallback")
                return self._fallback_retrieve(query, disaster_type)
            
            # Build where clause for filtering
            where_clause = {}
            if disaster_type:
                where_clause["disaster_type"] = disaster_type
            if content_types:
                where_clause["content_type"] = {"$in": content_types}
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            context_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    context_doc = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'][0] else 0.0,
                        'relevance_score': 1.0 - (results['distances'][0][i] if results['distances'][0] else 0.0)
                    }
                    context_docs.append(context_doc)
            
            return context_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return self._fallback_retrieve(query, disaster_type)
    
    def _fallback_retrieve(self, query: str, disaster_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback context retrieval when ChromaDB is not available"""
        context_docs = []
        
        # Simple keyword matching fallback
        query_lower = query.lower()
        
        # Search in procedures
        if 'procedures' in self.knowledge_base:
            procedures = self.knowledge_base['procedures']
            
            # If disaster type specified, prioritize that
            if disaster_type and disaster_type.lower() in procedures:
                disaster_procedures = procedures[disaster_type.lower()]
                for procedure in disaster_procedures[:3]:  # Top 3
                    context_docs.append({
                        'content': procedure,
                        'metadata': {'type': 'procedure', 'disaster_type': disaster_type.lower()},
                        'relevance_score': 0.8
                    })
            else:
                # Search all procedures for relevant keywords
                for d_type, procedure_list in procedures.items():
                    for procedure in procedure_list:
                        if any(word in procedure.lower() for word in query_lower.split()):
                            context_docs.append({
                                'content': procedure,
                                'metadata': {'type': 'procedure', 'disaster_type': d_type},
                                'relevance_score': 0.6
                            })
                            if len(context_docs) >= 5:
                                break
                    if len(context_docs) >= 5:
                        break
        
        return context_docs[:5]  # Return top 5
    
    def get_emergency_contacts(self, disaster_type: Optional[str] = None) -> Dict[str, str]:
        """Get relevant emergency contacts"""
        try:
            if 'procedures' in self.knowledge_base and 'contacts' in self.knowledge_base['procedures']:
                return self.knowledge_base['procedures']['contacts']
            
            # Fallback contacts
            return {
                'fire_department': '+1-555-FIRE-911',
                'police': '+1-555-POLICE-911',
                'medical_services': '+1-555-MEDICAL-911',
                'emergency_management': '+1-555-EMERGENCY'
            }
            
        except Exception as e:
            logger.error(f"Failed to get emergency contacts: {e}")
            return {}
    
    def get_resource_requirements(self, 
                                disaster_type: str, 
                                severity: float,
                                affected_population: int = 0) -> Dict[str, Any]:
        """Get resource requirements for specific emergency"""
        try:
            base_requirements = {
                'fire': {
                    'personnel': {'firefighters': max(5, int(severity)), 'paramedics': 2},
                    'vehicles': ['fire_engines', 'ambulances'],
                    'equipment': ['fire_suppression', 'medical_supplies']
                },
                'flood': {
                    'personnel': {'rescue_teams': max(3, int(severity/2)), 'coordinators': 2},
                    'vehicles': ['rescue_boats', 'emergency_vehicles'],
                    'equipment': ['life_jackets', 'communication_devices']
                },
                'earthquake': {
                    'personnel': {'search_rescue': max(8, int(severity)), 'medical': 4},
                    'vehicles': ['heavy_rescue', 'ambulances', 'command_vehicles'],
                    'equipment': ['search_equipment', 'medical_supplies', 'communication']
                }
            }
            
            # Get base requirements
            requirements = base_requirements.get(disaster_type.lower(), {
                'personnel': {'general_responders': max(3, int(severity))},
                'vehicles': ['emergency_vehicles'],
                'equipment': ['basic_equipment']
            })
            
            # Scale based on affected population
            if affected_population > 100:
                population_multiplier = min(affected_population / 100, 5.0)  # Cap at 5x
                
                # Scale personnel
                if 'personnel' in requirements:
                    for role, count in requirements['personnel'].items():
                        requirements['personnel'][role] = int(count * population_multiplier)
            
            return requirements
            
        except Exception as e:
            logger.error(f"Failed to get resource requirements: {e}")
            return {}
    
    def get_evacuation_procedures(self, disaster_type: str, location: Optional[Dict] = None) -> List[str]:
        """Get evacuation procedures for specific disaster type"""
        try:
            evacuation_procedures = {
                'fire': [
                    "Alert all occupants using fire alarm system",
                    "Evacuate using nearest safe exit, avoid elevators",
                    "Move to designated assembly point upwind from fire",
                    "Account for all personnel and report to incident commander",
                    "Do not re-enter building until cleared by fire department"
                ],
                'flood': [
                    "Move to higher ground immediately",
                    "Avoid walking or driving through flood water",
                    "Turn off utilities if time permits and it's safe to do so",
                    "Take emergency supplies and important documents",
                    "Follow designated evacuation routes to emergency shelters"
                ],
                'earthquake': [
                    "Drop, cover, and hold on during shaking",
                    "After shaking stops, evacuate if building is damaged",
                    "Use stairs, never elevators",
                    "Watch for hazards: broken glass, gas leaks, structural damage",
                    "Move to open areas away from buildings and power lines"
                ],
                'storm': [
                    "Seek shelter in sturdy building away from windows",
                    "If outdoors, lie flat in low-lying area",
                    "Avoid trees, poles, and metal objects",
                    "If in vehicle, pull over and stay inside",
                    "Wait for all-clear before moving to evacuation areas"
                ]
            }
            
            return evacuation_procedures.get(disaster_type.lower(), [
                "Follow emergency exit procedures",
                "Move to designated safe area",
                "Wait for further instructions from emergency personnel"
            ])
            
        except Exception as e:
            logger.error(f"Failed to get evacuation procedures: {e}")
            return []
    
    def add_knowledge(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Add new knowledge to the RAG system"""
        try:
            if not self.collection:
                logger.warning("Cannot add knowledge - ChromaDB not available")
                return False
            
            # Generate unique ID
            doc_id = f"custom_{metadata.get('type', 'unknown')}_{len(self.collection.get()['ids'])}"
            
            # Add to ChromaDB
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Added new knowledge document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return False
    
    def update_knowledge_base(self) -> bool:
        """Update knowledge base from files"""
        try:
            # Reload knowledge base
            self.knowledge_base = self._load_knowledge_base()
            
            # Clear and re-index
            if self.collection:
                # Reset collection
                self.chroma_client.delete_collection(name=self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Emergency response procedures and knowledge"}
                )
                
                # Re-index
                self._index_knowledge_base()
            
            logger.info("Knowledge base updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}")
            return False

# Usage example and testing
async def test_rag_system():
    """Test the RAG system"""
    rag = EmergencyRAGSystem()
    
    # Test context retrieval
    context = rag.retrieve_relevant_context(
        "building fire emergency response",
        disaster_type="fire",
        max_results=3
    )
    
    print("Retrieved Context:")
    for i, doc in enumerate(context):
        print(f"{i+1}. {doc['content']}")
        print(f"   Relevance: {doc['relevance_score']:.2f}")
        print(f"   Type: {doc['metadata'].get('type', 'unknown')}")
    
    # Test resource requirements
    resources = rag.get_resource_requirements("fire", severity=7, affected_population=200)
    print(f"\nResource Requirements: {resources}")
    
    # Test evacuation procedures
    evacuation = rag.get_evacuation_procedures("fire")
    print(f"\nEvacuation Procedures: {evacuation}")
    
    # Test emergency contacts
    contacts = rag.get_emergency_contacts()
    print(f"\nEmergency Contacts: {contacts}")

if __name__ == "__main__":
    asyncio.run(test_rag_system())