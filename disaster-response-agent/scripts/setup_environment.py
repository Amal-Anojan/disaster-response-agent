import os
import sys
import json
import logging
import subprocess
import platform
from pathlib import Path
from datetime import datetime
import shutil
import venv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Complete environment setup for Disaster Response Multi-Modal Agent"""
    
    def __init__(self):
        """Initialize environment setup"""
        self.base_path = Path(__file__).parent.parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.platform = platform.system().lower()
        
        # Setup paths
        self.venv_path = self.base_path / 'venv'
        self.data_path = self.base_path / 'data'
        self.logs_path = self.base_path / 'logs'
        self.config_path = self.base_path / 'config'
        
        # Required Python version
        self.min_python_version = (3, 9)
        self.recommended_python_version = (3, 11)
        
    def check_system_requirements(self) -> bool:
        """Check system requirements"""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        current_version = sys.version_info[:2]
        if current_version < self.min_python_version:
            logger.error(f"❌ Python {self.min_python_version[0]}.{self.min_python_version[1]}+ required. Current: {self.python_version}")
            return False
        
        if current_version < self.recommended_python_version:
            logger.warning(f"⚠️  Python {self.recommended_python_version[0]}.{self.recommended_python_version[1]}+ recommended. Current: {self.python_version}")
        else:
            logger.info(f"✅ Python version: {self.python_version}")
        
        # Check available disk space
        free_space_gb = shutil.disk_usage(self.base_path).free / (1024**3)
        if free_space_gb < 5:
            logger.error(f"❌ Insufficient disk space. Available: {free_space_gb:.1f}GB, Required: 5GB+")
            return False
        else:
            logger.info(f"✅ Disk space: {free_space_gb:.1f}GB available")
        
        # Check if pip is available
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                         capture_output=True, check=True)
            logger.info("✅ pip is available")
        except subprocess.CalledProcessError:
            logger.error("❌ pip is not available. Please install pip.")
            return False
        
        return True
    
    def create_directory_structure(self):
        """Create required directory structure"""
        logger.info("📁 Creating directory structure...")
        
        directories = [
            'data/datasets/sample_images',
            'data/knowledge_base',
            'data/temp_images',
            'logs',
            'config',
            'tests',
            'scripts',
            'docs'
        ]
        
        for directory in directories:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📂 Created: {directory}/")
    
    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment"""
        logger.info("🐍 Creating Python virtual environment...")
        
        if self.venv_path.exists():
            logger.info("Virtual environment already exists. Removing old one...")
            shutil.rmtree(self.venv_path)
        
        try:
            # Create virtual environment
            venv.create(self.venv_path, with_pip=True)
            logger.info(f"✅ Virtual environment created at: {self.venv_path}")
            
            # Get python executable path in venv
            if self.platform == 'windows':
                python_exe = self.venv_path / 'Scripts' / 'python.exe'
                pip_exe = self.venv_path / 'Scripts' / 'pip.exe'
            else:
                python_exe = self.venv_path / 'bin' / 'python'
                pip_exe = self.venv_path / 'bin' / 'pip'
            
            # Verify virtual environment
            result = subprocess.run([str(python_exe), '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Virtual environment Python: {result.stdout.strip()}")
                return True
            else:
                logger.error("❌ Failed to verify virtual environment")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        # Get pip executable path
        if self.platform == 'windows':
            pip_exe = self.venv_path / 'Scripts' / 'pip.exe'
        else:
            pip_exe = self.venv_path / 'bin' / 'pip'
        
        requirements_file = self.base_path / 'requirements.txt'
        
        if not requirements_file.exists():
            logger.error(f"❌ requirements.txt not found at {requirements_file}")
            return False
        
        try:
            # Upgrade pip first
            logger.info("⬆️  Upgrading pip...")
            subprocess.run([str(pip_exe), 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            
            # Install dependencies
            logger.info("📥 Installing requirements...")
            result = subprocess.run([str(pip_exe), 'install', '-r', str(requirements_file)], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ All dependencies installed successfully")
                
                # Show installed packages
                installed_result = subprocess.run([str(pip_exe), 'list'], 
                                                capture_output=True, text=True)
                package_count = len(installed_result.stdout.strip().split('\n')) - 2  # Subtract header lines
                logger.info(f"📊 Total packages installed: {package_count}")
                
                return True
            else:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_environment_file(self):
        """Setup environment configuration file"""
        logger.info("⚙️  Setting up environment configuration...")
        
        env_example_path = self.base_path / '.env.example'
        env_path = self.base_path / '.env'
        
        if env_path.exists():
            logger.info("Environment file already exists. Keeping existing configuration.")
            return
        
        if env_example_path.exists():
            # Copy example to .env
            shutil.copy(env_example_path, env_path)
            logger.info("✅ Created .env from .env.example")
        else:
            # Create basic .env file
            env_content = """# Disaster Response Multi-Modal Agent Configuration

# Required API Keys (Get free keys from the respective services)
CEREBRAS_API_KEY=your_cerebras_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Security
SECRET_KEY=generate_a_secure_random_key_here

# Application Settings
HOST=0.0.0.0
PORT=8080
DEBUG=false
ENVIRONMENT=development

# Database (SQLite by default)
DATABASE_URL=sqlite:///data/disaster_response.db

# Logging
LOG_LEVEL=INFO
LOG_TO_CONSOLE=true
LOG_TO_FILE=true

# Features
ENABLE_VISION_ANALYSIS=true
ENABLE_TEXT_ANALYSIS=true
ENABLE_RESOURCE_ALLOCATION=true
"""
            
            with open(env_path, 'w') as f:
                f.write(env_content)
            
            logger.info("✅ Created basic .env file")
        
        logger.warning("⚠️  Please edit .env file and add your API keys!")
    
    def initialize_database(self) -> bool:
        """Initialize database and load sample data"""
        logger.info("🗄️  Initializing database...")
        
        try:
            # Get python executable path
            if self.platform == 'windows':
                python_exe = self.venv_path / 'Scripts' / 'python.exe'
            else:
                python_exe = self.venv_path / 'bin' / 'python'
            
            # Run database initialization
            scripts_path = self.base_path / 'scripts'
            load_script = scripts_path / 'load_demo_data.py'
            
            if load_script.exists():
                result = subprocess.run([str(python_exe), str(load_script)], 
                                      capture_output=True, text=True, 
                                      cwd=str(self.base_path))
                
                if result.returncode == 0:
                    logger.info("✅ Database initialized with sample data")
                    return True
                else:
                    logger.warning(f"⚠️  Database initialization had issues: {result.stderr}")
                    return False
            else:
                logger.warning("⚠️  load_demo_data.py script not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            return False
    
    def create_startup_scripts(self):
        """Create startup scripts for easy launching"""
        logger.info("🚀 Creating startup scripts...")
        
        # Create start script for Unix/Linux/Mac
        if self.platform != 'windows':
            start_script_content = f"""#!/bin/bash
# Disaster Response Multi-Modal Agent Startup Script

echo "🚀 Starting Disaster Response Multi-Modal Agent..."

# Change to project directory
cd "{self.base_path}"

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists and has API keys
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run setup_environment.py first."
    exit 1
fi

# Check for API keys
if ! grep -q "CEREBRAS_API_KEY=your_cerebras" .env && ! grep -q "GEMINI_API_KEY=your_gemini" .env; then
    echo "✅ Environment configured"
else
    echo "⚠️  WARNING: Please set your API keys in .env file!"
    echo "   Edit .env and replace 'your_cerebras_api_key_here' and 'your_gemini_api_key_here'"
    echo "   with your actual API keys from:"
    echo "   - Cerebras: https://cloud.cerebras.ai/"
    echo "   - Gemini: https://makersuite.google.com/"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start the application
echo "🎯 Launching main application..."
python main.py

echo "👋 Disaster Response System stopped."
"""
            
            start_script_path = self.base_path / 'start.sh'
            with open(start_script_path, 'w') as f:
                f.write(start_script_content)
            
            # Make executable
            os.chmod(start_script_path, 0o755)
            logger.info("✅ Created start.sh (Unix/Linux/Mac)")
        
        # Create start script for Windows
        start_bat_content = f"""@echo off
REM Disaster Response Multi-Modal Agent Startup Script

echo 🚀 Starting Disaster Response Multi-Modal Agent...

REM Change to project directory
cd /d "{self.base_path}"

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Check if .env file exists
if not exist .env (
    echo ❌ .env file not found. Please run setup_environment.py first.
    pause
    exit /b 1
)

REM Start the application
echo 🎯 Launching main application...
python main.py

echo 👋 Disaster Response System stopped.
pause
"""
        
        start_bat_path = self.base_path / 'start.bat'
        with open(start_bat_path, 'w') as f:
            f.write(start_bat_content)
        
        logger.info("✅ Created start.bat (Windows)")
    
    def create_health_check_script(self):
        """Create health check script"""
        logger.info("🏥 Creating health check script...")
        
        # Create health check script for Unix/Linux/Mac
        if self.platform != 'windows':
            health_script_content = f"""#!/bin/bash
# Disaster Response Health Check Script

cd "{self.base_path}"
source venv/bin/activate

echo "🏥 Running health check..."
python scripts/health_check.py

echo "Health check completed."
"""
            
            health_script_path = self.base_path / 'health_check.sh'
            with open(health_script_path, 'w') as f:
                f.write(health_script_content)
            
            os.chmod(health_script_path, 0o755)
            logger.info("✅ Created health_check.sh")
        
        # Create health check script for Windows
        health_bat_content = f"""@echo off
cd /d "{self.base_path}"
call venv\\Scripts\\activate.bat

echo 🏥 Running health check...
python scripts\\health_check.py

pause
"""
        
        health_bat_path = self.base_path / 'health_check.bat'
        with open(health_bat_path, 'w') as f:
            f.write(health_bat_content)
        
        logger.info("✅ Created health_check.bat")
    
    def check_api_services(self):
        """Check API service availability and provide setup instructions"""
        logger.info("🔗 Checking API service setup...")
        
        instructions = '''
📋 API SERVICE SETUP INSTRUCTIONS
═══════════════════════════════════════════════════════════════

🧠 CEREBRAS API (Vision Analysis)
────────────────────────────────────
1. Visit: https://cloud.cerebras.ai/
2. Create free account
3. Navigate to API Keys section
4. Generate new API key
5. Copy key to .env file (CEREBRAS_API_KEY)

💬 GOOGLE GEMINI API (Language Model)  
──────────────────────────────────────
1. Visit: https://makersuite.google.com/
2. Sign in with Google account
3. Click "Get API Key"
4. Create new API key
5. Copy key to .env file (GEMINI_API_KEY)

💡 BOTH SERVICES OFFER GENEROUS FREE TIERS!
═══════════════════════════════════════════════════════════════

After getting your API keys:
1. Edit the .env file in this directory
2. Replace the placeholder values with your actual keys
3. Run: ./start.sh (Linux/Mac) or start.bat (Windows)
        '''
        
        print(instructions)
        
        # Save instructions to file
        instructions_file = self.base_path / 'API_SETUP_INSTRUCTIONS.txt'
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        logger.info("💾 API setup instructions saved to API_SETUP_INSTRUCTIONS.txt")
    
    def run_complete_setup(self) -> bool:
        """Run complete environment setup"""
        logger.info("🚀 Starting complete environment setup...")
        
        setup_start_time = datetime.now()
        
        try:
            # Step 1: Check system requirements
            if not self.check_system_requirements():
                return False
            
            # Step 2: Create directory structure
            self.create_directory_structure()
            
            # Step 3: Create virtual environment
            if not self.create_virtual_environment():
                return False
            
            # Step 4: Install dependencies
            if not self.install_dependencies():
                return False
            
            # Step 5: Setup environment file
            self.setup_environment_file()
            
            # Step 6: Create startup scripts
            self.create_startup_scripts()
            
            # Step 7: Create health check script
            self.create_health_check_script()
            
            # Step 8: Initialize database (optional, may fail without API keys)
            database_ok = self.initialize_database()
            
            # Step 9: Show API setup instructions
            self.check_api_services()
            
            # Calculate setup time
            setup_time = (datetime.now() - setup_start_time).total_seconds()
            
            # Final summary
            print("\n" + "="*70)
            print("🎉 ENVIRONMENT SETUP COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"⏱️  Setup time: {setup_time:.1f} seconds")
            print(f"🐍 Python version: {self.python_version}")
            print(f"💻 Platform: {platform.system()} {platform.release()}")
            print(f"📂 Project directory: {self.base_path}")
            
            print("\n📋 NEXT STEPS:")
            print("1. 🔑 Get your FREE API keys:")
            print("   • Cerebras: https://cloud.cerebras.ai/")
            print("   • Gemini: https://makersuite.google.com/")
            print("2. ✏️  Edit .env file and add your API keys")
            print("3. 🚀 Run the system:")
            if self.platform == 'windows':
                print("   • Windows: Double-click start.bat")
            else:
                print("   • Linux/Mac: ./start.sh")
            print("4. 🌐 Open dashboard: http://localhost:8501")
            print("5. 📊 API docs: http://localhost:8080/docs")
            
            if not database_ok:
                print("\n⚠️  Database initialization incomplete - add API keys first!")
            
            print("\n💡 Need help? Check API_SETUP_INSTRUCTIONS.txt")
            print("="*70)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_requirements_if_missing(self):
        """Create requirements.txt if it doesn't exist"""
        requirements_file = self.base_path / 'requirements.txt'
        
        if not requirements_file.exists():
            logger.info("📝 Creating requirements.txt file...")
            
            requirements_content = """# Disaster Response Multi-Modal Agent Dependencies

# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
streamlit==1.28.1

# AI and ML
google-generativeai==0.3.1
requests==2.31.0
pillow==10.0.1
numpy==1.24.3
pandas==2.0.3

# Database
sqlalchemy==2.0.23
alembic==1.13.0

# Data Validation
pydantic==2.5.0
email-validator==2.1.0

# Authentication & Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Utilities
python-dotenv==1.0.0
aiofiles==23.2.1
aiohttp==3.9.1
asyncio==3.4.3

# Visualization and UI
plotly==5.17.0
folium==0.15.0
streamlit-folium==0.15.0

# Image Processing
opencv-python==4.8.1.78
scipy==1.11.4

# Development and Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Monitoring and Logging
structlog==23.2.0

# Optional: Database drivers
psycopg2-binary==2.9.9  # PostgreSQL
redis==5.0.1  # Redis cache
"""
            
            with open(requirements_file, 'w') as f:
                f.write(requirements_content.strip())
            
            logger.info("✅ Created requirements.txt")

def main():
    """Main setup function"""
    print("🚀 Disaster Response Multi-Modal Agent - Environment Setup")
    print("=" * 60)
    
    try:
        setup = EnvironmentSetup()
        
        # Create requirements.txt if missing
        setup.create_requirements_if_missing()
        
        # Run complete setup
        success = setup.run_complete_setup()
        
        if success:
            print("\n🎉 Setup completed successfully!")
            print("Ready to build your AI-powered emergency response system! 🆘🤖")
            return 0
        else:
            print("\n❌ Setup failed. Please check the error messages above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)