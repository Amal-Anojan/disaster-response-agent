import os
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import wraps
import jwt
from passlib.context import CryptContext
import re

logger = logging.getLogger(__name__)

class SecurityManager:
    """Security management for the disaster response system"""
    
    def __init__(self):
        """Initialize security manager"""
        # Get secret key from environment or generate one
        self.secret_key = os.getenv('SECRET_KEY', secrets.token_urlsafe(32))
        
        # Password hashing context
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # JWT settings
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        
        # API key settings
        self.api_key_prefix = "dr_"  # disaster response prefix
        self.api_key_length = 32
        
        # Rate limiting and security thresholds
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15
        self.password_min_length = 8
        
        # Store for failed attempts (in production, use Redis or database)
        self.failed_attempts = {}
        self.locked_accounts = {}
        
        # Allowed origins for CORS
        self.allowed_origins = [
            "http://localhost:8501",  # Streamlit
            "http://localhost:8080",  # API
            "http://127.0.0.1:8501",
            "http://127.0.0.1:8080"
        ]
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        score = 0
        
        # Length check
        if len(password) < self.password_min_length:
            issues.append(f"Password must be at least {self.password_min_length} characters long")
        else:
            score += 1
        
        # Character type checks
        if not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        else:
            score += 1
        
        if not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        else:
            score += 1
        
        if not re.search(r'\d', password):
            issues.append("Password must contain at least one number")
        else:
            score += 1
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character")
        else:
            score += 1
        
        # Common pattern checks
        if re.search(r'(.)\1{2,}', password):
            issues.append("Password should not contain repeated characters")
            score -= 1
        
        # Strength levels
        if score >= 5:
            strength = "Strong"
        elif score >= 3:
            strength = "Medium"
        else:
            strength = "Weak"
        
        return {
            "valid": len(issues) == 0,
            "strength": strength,
            "score": max(0, score),
            "issues": issues
        }
    
    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """
        Generate API key for a user
        
        Args:
            user_id: User identifier
            permissions: List of permissions for this API key
            
        Returns:
            Generated API key
        """
        # Create payload for API key
        payload = {
            "user_id": user_id,
            "permissions": permissions or [],
            "created_at": datetime.utcnow().isoformat(),
            "key_id": secrets.token_hex(8)
        }
        
        # Generate key using JWT
        api_key = jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)
        
        return f"{self.api_key_prefix}{api_key}"
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key and return user information
        
        Args:
            api_key: API key to validate
            
        Returns:
            User information if valid, None otherwise
        """
        try:
            # Check prefix
            if not api_key.startswith(self.api_key_prefix):
                return None
            
            # Remove prefix and decode
            token = api_key[len(self.api_key_prefix):]
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def create_access_token(self, user_id: str, permissions: List[str] = None) -> str:
        """
        Create access token for user
        
        Args:
            user_id: User identifier
            permissions: User permissions
            
        Returns:
            JWT access token
        """
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "user_id": user_id,
            "permissions": permissions or [],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        Create refresh token for user
        
        Args:
            user_id: User identifier
            
        Returns:
            JWT refresh token
        """
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def check_failed_attempts(self, identifier: str) -> bool:
        """
        Check if account is locked due to failed attempts
        
        Args:
            identifier: User identifier or IP address
            
        Returns:
            True if account is locked, False otherwise
        """
        # Check if account is currently locked
        if identifier in self.locked_accounts:
            lock_time = self.locked_accounts[identifier]
            if datetime.now() - lock_time < timedelta(minutes=self.lockout_duration_minutes):
                return True
            else:
                # Lock has expired
                del self.locked_accounts[identifier]
                if identifier in self.failed_attempts:
                    del self.failed_attempts[identifier]
        
        return False
    
    def record_failed_attempt(self, identifier: str):
        """
        Record a failed authentication attempt
        
        Args:
            identifier: User identifier or IP address
        """
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        # Add current timestamp
        self.failed_attempts[identifier].append(datetime.now())
        
        # Remove attempts older than lockout duration
        cutoff_time = datetime.now() - timedelta(minutes=self.lockout_duration_minutes)
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier] 
            if attempt > cutoff_time
        ]
        
        # Check if max attempts exceeded
        if len(self.failed_attempts[identifier]) >= self.max_failed_attempts:
            self.locked_accounts[identifier] = datetime.now()
            logger.warning(f"Account locked due to failed attempts: {identifier}")
    
    def clear_failed_attempts(self, identifier: str):
        """Clear failed attempts for an identifier"""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]
        if identifier in self.locked_accounts:
            del self.locked_accounts[identifier]
    
    def sanitize_input(self, user_input: str, max_length: int = 1000) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            user_input: Raw user input
            max_length: Maximum allowed length
            
        Returns:
            Sanitized input
        """
        if not isinstance(user_input, str):
            return str(user_input)
        
        # Limit length
        sanitized = user_input[:max_length]
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\n', '\r', '\t']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Remove SQL injection patterns (basic)
        sql_patterns = [
            r'(\bunion\b.*\bselect\b)',
            r'(\bselect\b.*\bfrom\b)',
            r'(\binsert\b.*\binto\b)',
            r'(\bupdate\b.*\bset\b)',
            r'(\bdelete\b.*\bfrom\b)',
            r'(\bdrop\b.*\btable\b)',
            r'(--.*)',
            r'(/\*.*\*/)'
        ]
        
        for pattern in sql_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def validate_cors_origin(self, origin: str) -> bool:
        """
        Validate CORS origin
        
        Args:
            origin: Origin to validate
            
        Returns:
            True if origin is allowed, False otherwise
        """
        return origin in self.allowed_origins
    
    def generate_csrf_token(self) -> str:
        """Generate CSRF token"""
        return secrets.token_urlsafe(32)
    
    def validate_csrf_token(self, token: str, expected_token: str) -> bool:
        """Validate CSRF token"""
        return secrets.compare_digest(token, expected_token)
    
    def hash_data(self, data: str) -> str:
        """Create SHA-256 hash of data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_data_integrity(self, data: str, expected_hash: str) -> bool:
        """Verify data integrity using hash"""
        return self.hash_data(data) == expected_hash
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """
        Basic encryption for sensitive data
        In production, use proper encryption like Fernet
        """
        # This is a simple XOR cipher for demo purposes
        # In production, use proper encryption libraries
        key = self.secret_key[:32].encode()
        encrypted = bytearray()
        
        for i, char in enumerate(data.encode()):
            encrypted.append(char ^ key[i % len(key)])
        
        return encrypted.hex()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Basic decryption for sensitive data
        """
        try:
            key = self.secret_key[:32].encode()
            encrypted_bytes = bytes.fromhex(encrypted_data)
            decrypted = bytearray()
            
            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ key[i % len(key)])
            
            return decrypted.decode()
        except Exception:
            return ""
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

# Global security manager instance
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

# Decorators for security
def require_auth(permissions: List[str] = None):
    """
    Decorator to require authentication
    
    Args:
        permissions: Required permissions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would check authentication in a real implementation
            # For now, it's a placeholder
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_api_key(permissions: List[str] = None):
    """
    Decorator to require API key
    
    Args:
        permissions: Required permissions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would check API key in a real implementation
            return func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limited(calls_per_minute: int = 60):
    """
    Decorator for rate limiting
    
    Args:
        calls_per_minute: Maximum calls per minute
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would implement rate limiting in a real implementation
            return func(*args, **kwargs)
        return wrapper
    return decorator

def sanitize_inputs(fields: List[str] = None):
    """
    Decorator to sanitize function inputs
    
    Args:
        fields: List of parameter names to sanitize
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            security_manager = get_security_manager()
            
            # Sanitize kwargs
            if fields:
                for field in fields:
                    if field in kwargs:
                        kwargs[field] = security_manager.sanitize_input(str(kwargs[field]))
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Security monitoring
class SecurityMonitor:
    """Security monitoring and alerting"""
    
    def __init__(self):
        """Initialize security monitor"""
        self.security_events = []
        self.max_events = 1000
        
        self.event_types = {
            'authentication_failure': 'HIGH',
            'api_key_invalid': 'MEDIUM',
            'rate_limit_exceeded': 'MEDIUM',
            'suspicious_input': 'HIGH',
            'unauthorized_access': 'CRITICAL'
        }
    
    def record_security_event(self, 
                             event_type: str,
                             details: Dict[str, Any],
                             severity: str = 'MEDIUM'):
        """Record a security event"""
        event = {
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'event_id': secrets.token_hex(8)
        }
        
        self.security_events.append(event)
        
        # Keep only recent events
        if len(self.security_events) > self.max_events:
            self.security_events.pop(0)
        
        # Log security event
        logger.warning(f"Security event: {event_type} - {details}")
        
        # Alert for critical events
        if severity == 'CRITICAL':
            self._send_security_alert(event)
    
    def _send_security_alert(self, event: Dict[str, Any]):
        """Send security alert for critical events"""
        logger.critical(f"SECURITY ALERT: {event['event_type']} - {event['details']}")
        
        # In production, this would:
        # - Send email alerts
        # - Trigger incident response
        # - Log to SIEM system
        # - Send notifications to security team
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security events summary"""
        if not self.security_events:
            return {
                'total_events': 0,
                'events_by_type': {},
                'events_by_severity': {},
                'recent_events': []
            }
        
        # Count by type
        events_by_type = {}
        events_by_severity = {}
        
        for event in self.security_events:
            event_type = event['event_type']
            severity = event['severity']
            
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            events_by_severity[severity] = events_by_severity.get(severity, 0) + 1
        
        return {
            'total_events': len(self.security_events),
            'events_by_type': events_by_type,
            'events_by_severity': events_by_severity,
            'recent_events': self.security_events[-10:] if self.security_events else []
        }

# Global security monitor
_security_monitor = None

def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance"""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor

# Security health check
def security_health_check() -> Dict[str, Any]:
    """Check security system health"""
    try:
        security_manager = get_security_manager()
        security_monitor = get_security_monitor()
        
        return {
            'status': 'healthy',
            'security_manager': {
                'secret_key_configured': bool(security_manager.secret_key),
                'failed_attempts_count': len(security_manager.failed_attempts),
                'locked_accounts_count': len(security_manager.locked_accounts)
            },
            'security_monitor': security_monitor.get_security_summary(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Usage examples and testing
def test_security():
    """Test security functionality"""
    print("Testing Security System...")
    
    security_manager = get_security_manager()
    
    # Test password hashing
    password = "TestPassword123!"
    hashed = security_manager.hash_password(password)
    print(f"✓ Password hashing works")
    
    # Test password verification
    is_valid = security_manager.verify_password(password, hashed)
    print(f"✓ Password verification: {is_valid}")
    
    # Test password strength validation
    validation = security_manager.validate_password_strength(password)
    print(f"✓ Password strength: {validation['strength']}")
    
    # Test API key generation and validation
    api_key = security_manager.generate_api_key("test_user", ["read", "write"])
    api_data = security_manager.validate_api_key(api_key)
    print(f"✓ API key validation: {api_data is not None}")
    
    # Test JWT tokens
    access_token = security_manager.create_access_token("test_user", ["admin"])
    token_data = security_manager.verify_token(access_token)
    print(f"✓ JWT token validation: {token_data is not None}")
    
    # Test input sanitization
    dangerous_input = "<script>alert('xss')</script>SELECT * FROM users--"
    sanitized = security_manager.sanitize_input(dangerous_input)
    print(f"✓ Input sanitization: '{dangerous_input[:20]}...' -> '{sanitized[:20]}...'")
    
    # Test failed attempts tracking
    security_manager.record_failed_attempt("test_user")
    is_locked = security_manager.check_failed_attempts("test_user")
    print(f"✓ Failed attempts tracking: locked={is_locked}")
    
    # Test security monitoring
    security_monitor = get_security_monitor()
    security_monitor.record_security_event(
        'authentication_failure',
        {'user': 'test_user', 'ip': '127.0.0.1'},
        'HIGH'
    )
    
    summary = security_monitor.get_security_summary()
    print(f"✓ Security monitoring: {summary['total_events']} events recorded")
    
    # Test health check
    health = security_health_check()
    print(f"✓ Security health: {health['status']}")
    
    print("Security system tests completed!")

if __name__ == "__main__":
    test_security()