import asyncio
import aiohttp
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checker for disaster response system"""
    
    def __init__(self):
        """Initialize health checker"""
        self.base_path = Path(__file__).parent.parent
        self.results = {}
        self.start_time = None
        
        # Service endpoints
        self.endpoints = {
            'mcp_gateway': 'http://localhost:8080',
            'streamlit_dashboard': 'http://localhost:8501',
            'vision_service': 'http://localhost:8001',
            'llm_service': 'http://localhost:8002'
        }
        
        # Health check configuration
        self.timeout = 10
        self.max_retries = 3
        
    async def check_service_health(self, service_name: str, url: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        logger.info(f"Checking {service_name} at {url}")
        
        health_result = {
            'service': service_name,
            'url': url,
            'status': 'unknown',
            'response_time_ms': None,
            'error': None,
            'details': {}
        }
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                # Try health endpoint first
                health_endpoints = ['/health', '/healthz', '/ping', '/_health']
                
                for health_path in health_endpoints:
                    try:
                        async with session.get(
                            f"{url}{health_path}",
                            timeout=aiohttp.ClientTimeout(total=self.timeout)
                        ) as response:
                            response_time = (time.time() - start_time) * 1000
                            
                            if response.status == 200:
                                health_result.update({
                                    'status': 'healthy',
                                    'response_time_ms': round(response_time, 2),
                                    'http_status': response.status
                                })
                                
                                # Try to get detailed health info
                                try:
                                    health_data = await response.json()
                                    health_result['details'] = health_data
                                except:
                                    health_result['details'] = {'text_response': await response.text()}
                                
                                return health_result
                                
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        continue
                
                # If no health endpoint worked, try root endpoint
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status < 500:  # Any non-server error is considered "responding"
                            health_result.update({
                                'status': 'responding',
                                'response_time_ms': round(response_time, 2),
                                'http_status': response.status,
                                'details': {'note': 'Root endpoint responding but no health endpoint found'}
                            })
                        else:
                            health_result.update({
                                'status': 'unhealthy',
                                'response_time_ms': round(response_time, 2),
                                'http_status': response.status,
                                'error': f'Server error: {response.status}'
                            })
                            
                except asyncio.TimeoutError:
                    health_result.update({
                        'status': 'timeout',
                        'error': f'Service did not respond within {self.timeout} seconds'
                    })
                except Exception as e:
                    health_result.update({
                        'status': 'error',
                        'error': f'Connection error: {str(e)}'
                    })
                    
        except Exception as e:
            health_result.update({
                'status': 'error',
                'error': f'Failed to check service: {str(e)}'
            })
        
        return health_result
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        logger.info("Checking database health")
        
        db_result = {
            'component': 'database',
            'status': 'unknown',
            'details': {},
            'error': None
        }
        
        try:
            from models.database import health_check
            health_info = await health_check()
            
            db_result.update({
                'status': health_info.get('status', 'unknown'),
                'details': health_info
            })
            
        except ImportError:
            db_result.update({
                'status': 'module_missing',
                'error': 'Database module not available'
            })
        except Exception as e:
            db_result.update({
                'status': 'error',
                'error': str(e)
            })
        
        return db_result
    
    async def check_ai_services(self) -> Dict[str, Any]:
        """Check AI service configurations"""
        logger.info("Checking AI service configurations")
        
        ai_result = {
            'component': 'ai_services',
            'cerebras': {'status': 'unknown'},
            'gemini': {'status': 'unknown'},
            'overall_status': 'unknown'
        }
        
        # Check API keys
        cerebras_key = os.getenv('CEREBRAS_API_KEY')
        gemini_key = os.getenv('GEMINI_API_KEY')
        
        ai_result['cerebras'].update({
            'api_key_configured': bool(cerebras_key and len(cerebras_key) > 10),
            'status': 'configured' if cerebras_key and len(cerebras_key) > 10 else 'not_configured'
        })
        
        ai_result['gemini'].update({
            'api_key_configured': bool(gemini_key and len(gemini_key) > 10),
            'status': 'configured' if gemini_key and len(gemini_key) > 10 else 'not_configured'
        })
        
        # Test AI services if possible
        try:
            from vision_service.damage_analyzer import CerebrasVisionAnalyzer
            analyzer = CerebrasVisionAnalyzer()
            ai_result['cerebras']['module_available'] = True
        except Exception as e:
            ai_result['cerebras']['module_available'] = False
            ai_result['cerebras']['error'] = str(e)
        
        try:
            from llm_service.action_generator import EmergencyActionGenerator
            generator = EmergencyActionGenerator()
            ai_result['gemini']['module_available'] = True
        except Exception as e:
            ai_result['gemini']['module_available'] = False
            ai_result['gemini']['error'] = str(e)
        
        # Determine overall status
        cerebras_ok = ai_result['cerebras']['status'] == 'configured'
        gemini_ok = ai_result['gemini']['status'] == 'configured'
        
        if cerebras_ok and gemini_ok:
            ai_result['overall_status'] = 'healthy'
        elif cerebras_ok or gemini_ok:
            ai_result['overall_status'] = 'partial'
        else:
            ai_result['overall_status'] = 'unhealthy'
        
        return ai_result
    
    async def check_file_system(self) -> Dict[str, Any]:
        """Check file system and data availability"""
        logger.info("Checking file system health")
        
        fs_result = {
            'component': 'file_system',
            'status': 'unknown',
            'paths': {},
            'permissions': {}
        }
        
        # Check important paths
        important_paths = {
            'base_path': self.base_path,
            'data_path': self.base_path / 'data',
            'logs_path': self.base_path / 'logs',
            'config_path': self.base_path / 'config',
            'src_path': self.base_path / 'src'
        }
        
        all_paths_ok = True
        
        for name, path in important_paths.items():
            path_info = {
                'exists': path.exists(),
                'readable': False,
                'writable': False,
                'path': str(path)
            }
            
            if path.exists():
                try:
                    path_info['readable'] = os.access(path, os.R_OK)
                    path_info['writable'] = os.access(path, os.W_OK)
                except Exception as e:
                    path_info['error'] = str(e)
            else:
                all_paths_ok = False
                # Try to create if it doesn't exist
                try:
                    if name in ['data_path', 'logs_path']:
                        path.mkdir(parents=True, exist_ok=True)
                        path_info['created'] = True
                        path_info['exists'] = True
                        path_info['readable'] = True
                        path_info['writable'] = True
                    else:
                        all_paths_ok = False
                except Exception as e:
                    path_info['creation_error'] = str(e)
            
            fs_result['paths'][name] = path_info
        
        fs_result['status'] = 'healthy' if all_paths_ok else 'issues'
        
        return fs_result
    
    async def check_environment(self) -> Dict[str, Any]:
        """Check environment configuration"""
        logger.info("Checking environment configuration")
        
        env_result = {
            'component': 'environment',
            'status': 'unknown',
            'python_version': sys.version,
            'platform': sys.platform,
            'environment_variables': {},
            'required_modules': {}
        }
        
        # Check important environment variables
        important_env_vars = [
            'CEREBRAS_API_KEY',
            'GEMINI_API_KEY',
            'SECRET_KEY',
            'DATABASE_URL',
            'HOST',
            'PORT'
        ]
        
        for var in important_env_vars:
            value = os.getenv(var)
            env_result['environment_variables'][var] = {
                'set': bool(value),
                'length': len(value) if value else 0
            }
        
        # Check required Python modules
        required_modules = [
            'fastapi',
            'streamlit',
            'requests',
            'sqlalchemy',
            'pydantic',
            'google.generativeai',
            'PIL',
            'numpy',
            'pandas',
            'plotly'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
                env_result['required_modules'][module] = {'available': True}
            except ImportError as e:
                env_result['required_modules'][module] = {
                    'available': False,
                    'error': str(e)
                }
        
        # Determine overall environment status
        missing_modules = sum(1 for mod in env_result['required_modules'].values() if not mod['available'])
        missing_env_vars = sum(1 for var in env_result['environment_variables'].values() if not var['set'])
        
        if missing_modules == 0 and missing_env_vars <= 2:  # Allow some optional env vars
            env_result['status'] = 'healthy'
        elif missing_modules <= 2:  # Some optional modules missing
            env_result['status'] = 'partial'
        else:
            env_result['status'] = 'issues'
        
        return env_result
    
    async def perform_integration_test(self) -> Dict[str, Any]:
        """Perform basic integration test"""
        logger.info("Performing integration test")
        
        integration_result = {
            'component': 'integration_test',
            'status': 'unknown',
            'tests': {}
        }
        
        # Test 1: Database connection
        try:
            from models.database import get_database_manager
            db_manager = get_database_manager()
            connection_ok = db_manager.check_connection()
            integration_result['tests']['database_connection'] = {
                'status': 'pass' if connection_ok else 'fail',
                'result': connection_ok
            }
        except Exception as e:
            integration_result['tests']['database_connection'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Test 2: Vision service import
        try:
            from vision_service.damage_analyzer import CerebrasVisionAnalyzer
            analyzer = CerebrasVisionAnalyzer()
            integration_result['tests']['vision_service_import'] = {'status': 'pass'}
        except Exception as e:
            integration_result['tests']['vision_service_import'] = {
                'status': 'fail',
                'error': str(e)
            }
        
        # Test 3: LLM service import
        try:
            from llm_service.action_generator import EmergencyActionGenerator
            generator = EmergencyActionGenerator()
            integration_result['tests']['llm_service_import'] = {'status': 'pass'}
        except Exception as e:
            integration_result['tests']['llm_service_import'] = {
                'status': 'fail',
                'error': str(e)
            }
        
        # Test 4: Pipeline import
        try:
            from orchestrator.disaster_pipeline import DisasterResponsePipeline
            pipeline = DisasterResponsePipeline()
            integration_result['tests']['pipeline_import'] = {'status': 'pass'}
        except Exception as e:
            integration_result['tests']['pipeline_import'] = {
                'status': 'fail',
                'error': str(e)
            }
        
        # Determine overall integration status
        passed_tests = sum(1 for test in integration_result['tests'].values() if test['status'] == 'pass')
        total_tests = len(integration_result['tests'])
        
        if passed_tests == total_tests:
            integration_result['status'] = 'healthy'
        elif passed_tests >= total_tests * 0.75:
            integration_result['status'] = 'partial'
        else:
            integration_result['status'] = 'unhealthy'
        
        integration_result['pass_rate'] = f"{passed_tests}/{total_tests}"
        
        return integration_result
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        logger.info("🚀 Starting comprehensive health check...")
        self.start_time = time.time()
        
        # Run all health checks concurrently
        tasks = []
        
        # Service health checks
        for service_name, url in self.endpoints.items():
            tasks.append(self.check_service_health(service_name, url))
        
        # Component health checks
        tasks.extend([
            self.check_database_health(),
            self.check_ai_services(),
            self.check_file_system(),
            self.check_environment(),
            self.perform_integration_test()
        ])
        
        # Execute all checks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        service_results = []
        component_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = {
                    'status': 'error',
                    'error': str(result),
                    'check_index': i
                }
                if i < len(self.endpoints):
                    service_results.append(error_result)
                else:
                    component_results.append(error_result)
            else:
                if i < len(self.endpoints):
                    service_results.append(result)
                else:
                    component_results.append(result)
        
        # Calculate overall health
        total_checks = len(service_results) + len(component_results)
        healthy_checks = 0
        
        for result in service_results + component_results:
            if result.get('status') in ['healthy', 'responding', 'pass']:
                healthy_checks += 1
            elif result.get('status') in ['partial']:
                healthy_checks += 0.5
        
        health_score = (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
        
        if health_score >= 90:
            overall_status = 'healthy'
        elif health_score >= 70:
            overall_status = 'partial'
        elif health_score >= 50:
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'
        
        # Compile final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'check_duration_seconds': round(time.time() - self.start_time, 2),
            'overall_status': overall_status,
            'health_score': round(health_score, 1),
            'services': service_results,
            'components': component_results,
            'summary': {
                'total_checks': total_checks,
                'healthy_checks': int(healthy_checks),
                'issues_found': total_checks - int(healthy_checks),
                'recommendations': self._generate_recommendations(service_results + component_results)
            }
        }
        
        return final_results
    
    def _generate_recommendations(self, all_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on health check results"""
        recommendations = []
        
        # Check for common issues
        for result in all_results:
            status = result.get('status', 'unknown')
            
            if status == 'not_configured':
                if 'cerebras' in str(result).lower():
                    recommendations.append("Set CEREBRAS_API_KEY environment variable")
                if 'gemini' in str(result).lower():
                    recommendations.append("Set GEMINI_API_KEY environment variable")
            
            elif status in ['timeout', 'error']:
                if result.get('service'):
                    recommendations.append(f"Check if {result['service']} is running")
                elif result.get('component'):
                    recommendations.append(f"Investigate {result['component']} issues")
            
            elif status == 'unhealthy':
                if result.get('component') == 'database':
                    recommendations.append("Check database connection and configuration")
                
            elif 'module_missing' in status:
                recommendations.append("Install missing Python dependencies: pip install -r requirements.txt")
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        # Add general recommendations if no specific issues found
        if not recommendations:
            recommendations.append("System appears healthy - no immediate actions needed")
        
        return recommendations
    
    def print_health_report(self, results: Dict[str, Any]):
        """Print formatted health report"""
        print("\n" + "="*80)
        print("🏥 DISASTER RESPONSE SYSTEM HEALTH CHECK REPORT")
        print("="*80)
        
        # Overall status
        status_icon = {
            'healthy': '✅',
            'partial': '⚠️ ',
            'degraded': '🔶',
            'unhealthy': '❌'
        }
        
        print(f"\n📊 OVERALL STATUS: {status_icon.get(results['overall_status'], '❓')} {results['overall_status'].upper()}")
        print(f"💯 HEALTH SCORE: {results['health_score']:.1f}%")
        print(f"⏱️  CHECK DURATION: {results['check_duration_seconds']}s")
        print(f"🕐 TIMESTAMP: {results['timestamp']}")
        
        # Service status
        print(f"\n🚀 SERVICES ({len(results['services'])} checked)")
        print("-" * 50)
        for service in results['services']:
            status = service.get('status', 'unknown')
            icon = '✅' if status in ['healthy', 'responding'] else '❌' if status in ['error', 'timeout'] else '⚠️'
            service_name = service.get('service', 'Unknown')
            response_time = f" ({service.get('response_time_ms', 0):.0f}ms)" if service.get('response_time_ms') else ""
            print(f"{icon} {service_name}: {status.upper()}{response_time}")
        
        # Component status  
        print(f"\n🔧 COMPONENTS ({len(results['components'])} checked)")
        print("-" * 50)
        for component in results['components']:
            status = component.get('status', 'unknown')
            icon = '✅' if status in ['healthy', 'pass'] else '❌' if status in ['error', 'unhealthy', 'fail'] else '⚠️'
            component_name = component.get('component', 'Unknown')
            print(f"{icon} {component_name}: {status.upper()}")
        
        # Issues and recommendations
        if results['summary']['issues_found'] > 0:
            print(f"\n⚠️  ISSUES FOUND: {results['summary']['issues_found']}")
            print("-" * 50)
            
            for service in results['services']:
                if service.get('status') not in ['healthy', 'responding']:
                    print(f"❌ {service.get('service', 'Unknown')}: {service.get('error', 'Unknown error')}")
            
            for component in results['components']:
                if component.get('status') not in ['healthy', 'pass']:
                    print(f"❌ {component.get('component', 'Unknown')}: {component.get('error', 'Issues detected')}")
        
        # Recommendations
        if results['summary']['recommendations']:
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 50)
            for i, rec in enumerate(results['summary']['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Quick start guide
        if results['overall_status'] != 'healthy':
            print(f"\n🚀 QUICK START GUIDE")
            print("-" * 50)
            print("1. Install dependencies: pip install -r requirements.txt")
            print("2. Set API keys in .env file")
            print("3. Start services: python main.py")
            print("4. Run health check again: python scripts/health_check.py")
        
        print("\n" + "="*80)
        
        return results['overall_status'] == 'healthy'

async def main():
    """Main health check function"""
    try:
        checker = HealthChecker()
        results = await checker.run_comprehensive_health_check()
        
        # Print report
        is_healthy = checker.print_health_report(results)
        
        # Save results to file
        health_file = checker.base_path / 'logs' / 'health_check.json'
        health_file.parent.mkdir(exist_ok=True)
        
        with open(health_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Health check results saved to: {health_file}")
        
        return is_healthy
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)