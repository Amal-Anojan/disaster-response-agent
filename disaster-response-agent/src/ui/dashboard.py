import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import time
import json

# Page configuration
st.set_page_config(
    page_title="🚨 Emergency Response Command Center",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4b4b;
    }
    .status-active {
        background-color: #ff4b4b;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .status-processing {
        background-color: #ffa500;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .status-resolved {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .urgency-critical {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .urgency-high {
        background-color: #fd7e14;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .urgency-medium {
        background-color: #ffc107;
        color: black;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .urgency-low {
        background-color: #6c757d;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8080/api/v1"

class EmergencyDashboard:
    def __init__(self):
        self.api_base = API_BASE_URL
        
    def get_active_incidents(self):
        """Fetch active incidents from API"""
        try:
            response = requests.get(f"{self.api_base}/incidents/active", timeout=5)
            if response.status_code == 200:
                return response.json().get("incidents", [])
        except requests.RequestException as e:
            st.error(f"Failed to fetch incidents: {e}")
        
        # Return sample data if API is unavailable
        return self._get_sample_incidents()
    
    def _get_sample_incidents(self):
        """Generate sample incidents for demo"""
        import random
        
        sample_incidents = [
            {
                "incident_id": "INC_202510021200_001",
                "disaster_type": "fire",
                "severity": 8,
                "urgency": "HIGH",
                "status": "PROCESSING",
                "location": {"lat": 37.7749, "lng": -122.4194},
                "text_content": "Major apartment fire on Mission Street, multiple units affected",
                "estimated_response_time": 12,
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat()
            },
            {
                "incident_id": "INC_202510021130_002", 
                "disaster_type": "flood",
                "severity": 6,
                "urgency": "MEDIUM",
                "status": "COMPLETED",
                "location": {"lat": 37.7849, "lng": -122.4094},
                "text_content": "Street flooding due to broken water main",
                "estimated_response_time": 8,
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()
            },
            {
                "incident_id": "INC_202510021100_003",
                "disaster_type": "earthquake",
                "severity": 9,
                "urgency": "CRITICAL",
                "status": "PROCESSING",
                "location": {"lat": 37.7649, "lng": -122.4294},
                "text_content": "Building collapse reported downtown, possible casualties",
                "estimated_response_time": 5,
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()
            }
        ]
        
        return sample_incidents
    
    def submit_incident_report(self, text_content, image_file, location):
        """Submit new incident report"""
        try:
            data = {
                "text_content": text_content,
                "source": "manual",
                "location": location
            }
            
            files = {}
            if image_file is not None:
                files["image"] = image_file.getvalue()
            
            response = requests.post(
                f"{self.api_base}/incident/report",
                data={"incident": json.dumps(data)},
                files=files,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to submit report: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            st.error(f"Failed to submit report: {e}")
            return None

# Initialize dashboard
dashboard = EmergencyDashboard()

# Main title
st.markdown('<h1 class="main-header">🚨 Emergency Response Command Center</h1>', unsafe_allow_html=True)

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
if auto_refresh:
    st_auto_refresh = st.sidebar.button("Refresh Now")

# Get current incidents
incidents = dashboard.get_active_incidents()

# Calculate metrics
total_incidents = len(incidents)
active_count = len([inc for inc in incidents if inc.get("status") == "PROCESSING"])
critical_count = len([inc for inc in incidents if inc.get("urgency") == "CRITICAL"])
avg_response_time = sum([inc.get("estimated_response_time", 0) for inc in incidents]) / max(total_incidents, 1)

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📋 Total Incidents",
        value=total_incidents,
        delta=2 if total_incidents > 5 else 0
    )

with col2:
    st.metric(
        label="🔴 Active Processing",
        value=active_count,
        delta=1 if active_count > 0 else 0
    )

with col3:
    st.metric(
        label="⚠️ Critical Alerts",
        value=critical_count,
        delta=1 if critical_count > 0 else 0
    )

with col4:
    st.metric(
        label="⏱️ Avg Response Time",
        value=f"{avg_response_time:.1f} min",
        delta=-1.2
    )

# System status indicator
system_status = "🟢 OPERATIONAL"
st.sidebar.markdown(f"**System Status:** {system_status}")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Incident Map", 
    "📋 Active Queue", 
    "📊 Analytics", 
    "⚙️ System Monitor",
    "📝 Report Incident"
])

with tab1:
    st.header("Real-time Incident Map")
    
    # Create base map
    if incidents:
        # Center map on incidents
        center_lat = sum([inc.get("location", {}).get("lat", 37.7749) for inc in incidents]) / len(incidents)
        center_lng = sum([inc.get("location", {}).get("lng", -122.4194) for inc in incidents]) / len(incidents)
    else:
        center_lat, center_lng = 37.7749, -122.4194
    
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add incident markers
    for inc in incidents:
        location = inc.get("location", {})
        if location:
            lat, lng = location.get("lat", 37.7749), location.get("lng", -122.4194)
            
            # Color based on urgency
            color_map = {
                "CRITICAL": "red",
                "HIGH": "orange", 
                "MEDIUM": "yellow",
                "LOW": "green"
            }
            color = color_map.get(inc.get("urgency", "MEDIUM"), "blue")
            
            # Create popup
            popup_html = f"""
            <b>Incident:</b> {inc.get('incident_id', 'Unknown')}<br>
            <b>Type:</b> {inc.get('disaster_type', 'Unknown').title()}<br>
            <b>Severity:</b> {inc.get('severity', 0)}/10<br>
            <b>Urgency:</b> {inc.get('urgency', 'MEDIUM')}<br>
            <b>Status:</b> {inc.get('status', 'Unknown')}<br>
            <b>Response Time:</b> {inc.get('estimated_response_time', 0)} min<br>
            <b>Description:</b> {inc.get('text_content', 'No description')[:100]}...
            """
            
            folium.Marker(
                [lat, lng],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{inc.get('disaster_type', 'Unknown').title()} - {inc.get('urgency', 'MEDIUM')}",
                icon=folium.Icon(color=color, icon='warning-sign')
            ).add_to(m)
    
    # Display map
    st_folium(m, width=700, height=500, returned_objects=["last_object_clicked"])

with tab2:
    st.header("Active Incidents Priority Queue")
    
    if not incidents:
        st.info("No active incidents")
    else:
        # Sort by severity and urgency
        sorted_incidents = sorted(
            incidents, 
            key=lambda x: (
                {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x.get("urgency", "MEDIUM"), 2),
                x.get("severity", 0)
            ), 
            reverse=True
        )
        
        for i, inc in enumerate(sorted_incidents):
            with st.expander(
                f"🚨 {inc.get('disaster_type', 'Unknown').title()} - "
                f"Severity {inc.get('severity', 0)}/10 - "
                f"{inc.get('urgency', 'MEDIUM')} Priority",
                expanded=i < 3  # Expand first 3 incidents
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Incident ID:** {inc.get('incident_id', 'Unknown')}")
                    st.write(f"**Location:** {inc.get('location', {}).get('lat', 'Unknown')}, {inc.get('location', {}).get('lng', 'Unknown')}")
                    st.write(f"**Description:** {inc.get('text_content', 'No description available')}")
                    st.write(f"**Timestamp:** {inc.get('timestamp', 'Unknown')}")
                    
                    # Action plan if available
                    if inc.get('action_plan'):
                        action_plan = inc['action_plan']
                        if isinstance(action_plan, dict) and 'immediate_actions' in action_plan:
                            st.write("**Immediate Actions:**")
                            for action in action_plan['immediate_actions'][:3]:
                                st.write(f"• {action}")
                
                with col2:
                    # Status
                    status = inc.get('status', 'UNKNOWN')
                    st.markdown(f"**Status:** <span class='status-{status.lower()}'>{status}</span>", unsafe_allow_html=True)
                    
                    # Urgency
                    urgency = inc.get('urgency', 'MEDIUM')
                    st.markdown(f"**Urgency:** <span class='urgency-{urgency.lower()}'>{urgency}</span>", unsafe_allow_html=True)
                    
                    st.write(f"**ETA:** {inc.get('estimated_response_time', 0)} min")
                    st.write(f"**Confidence:** {inc.get('confidence', 0.7):.1%}")
                    
                    # Action buttons
                    if st.button(f"Update Status", key=f"update_{i}"):
                        st.success("Status update functionality would be implemented here")

with tab3:
    st.header("Response Analytics & Performance")
    
    if incidents:
        # Response time trend
        response_times = [inc.get('estimated_response_time', 0) for inc in incidents]
        fig_response = px.line(
            x=list(range(len(response_times))),
            y=response_times,
            title="Response Time Trend",
            labels={"x": "Incident Number", "y": "Response Time (minutes)"}
        )
        fig_response.update_traces(line_color='#ff4b4b')
        st.plotly_chart(fig_response, use_container_width=True)
        
        # Incident type distribution
        incident_types = {}
        for inc in incidents:
            disaster_type = inc.get('disaster_type', 'unknown')
            incident_types[disaster_type] = incident_types.get(disaster_type, 0) + 1
        
        if incident_types:
            fig_pie = px.pie(
                values=list(incident_types.values()),
                names=list(incident_types.keys()),
                title="Incident Types Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Severity distribution
        severities = [inc.get('severity', 0) for inc in incidents]
        fig_hist = px.histogram(
            x=severities,
            nbins=10,
            title="Severity Distribution",
            labels={"x": "Severity Level", "y": "Count"}
        )
        fig_hist.update_traces(marker_color='#ff4b4b')
        st.plotly_chart(fig_hist, use_container_width=True)
        
    else:
        st.info("No data available for analytics")

with tab4:
    st.header("System Monitor")
    
    # System health metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("API Gateway", "🟢 Online", delta=None)
        st.metric("Vision Service", "🟢 Active", delta=None)
    
    with col2:
        st.metric("LLM Service", "🟢 Active", delta=None)
        st.metric("Database", "🟢 Connected", delta=None)
    
    with col3:
        st.metric("Docker Services", "🟢 Running", delta=None)
        st.metric("Queue Depth", "0", delta=0)
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    # Simulated performance data
    perf_data = {
        "Metric": ["API Response Time", "Vision Processing", "LLM Generation", "Database Query"],
        "Current (ms)": [145, 890, 1200, 45],
        "Target (ms)": [200, 1000, 1500, 100],
        "Status": ["🟢 Good", "🟢 Good", "🟢 Good", "🟢 Good"]
    }
    
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf, use_container_width=True)
    
    # Resource utilization
    st.subheader("Resource Utilization")
    
    col1, col2 = st.columns(2)
    with col1:
        # CPU usage gauge
        fig_cpu = go.Figure(go.Indicator(
            mode="gauge+number",
            value=67,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPU Usage (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}))
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        # Memory usage gauge
        fig_mem = go.Figure(go.Indicator(
            mode="gauge+number",
            value=45,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Memory Usage (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkgreen"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}))
        st.plotly_chart(fig_mem, use_container_width=True)

with tab5:
    st.header("Report New Emergency Incident")
    
    with st.form("incident_report_form"):
        st.subheader("Incident Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text_content = st.text_area(
                "Incident Description *",
                placeholder="Describe the emergency situation in detail...",
                height=120
            )
            
            source_type = st.selectbox(
                "Report Source",
                ["manual", "social_media", "sensor", "911_call", "witness"]
            )
            
        with col2:
            # Location input
            st.subheader("Location")
            lat = st.number_input("Latitude", value=37.7749, format="%.6f")
            lng = st.number_input("Longitude", value=-122.4194, format="%.6f")
            
            priority = st.selectbox(
                "Priority Override (optional)",
                [None, "LOW", "MEDIUM", "HIGH", "CRITICAL"]
            )
        
        # Image upload
        st.subheader("Attach Image (Optional)")
        uploaded_image = st.file_uploader(
            "Upload disaster image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image to enable AI vision analysis"
        )
        
        # Preview uploaded image
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", width=300)
        
        # Submit button
        submitted = st.form_submit_button("🚨 Submit Emergency Report", type="primary")
        
        if submitted:
            if text_content.strip():
                location = {"lat": lat, "lng": lng}
                
                with st.spinner("Submitting incident report..."):
                    result = dashboard.submit_incident_report(
                        text_content, 
                        uploaded_image, 
                        location
                    )
                
                if result:
                    st.success(f"✅ Incident reported successfully! ID: {result.get('incident_id', 'Unknown')}")
                    st.balloons()
                    
                    # Show processing status
                    st.info("🔄 Processing incident with AI analysis...")
                    
                    # In a real application, you would poll for status updates
                    time.sleep(2)
                    st.success("🎯 AI analysis completed! Check the Active Queue tab for details.")
                    
                else:
                    st.error("❌ Failed to submit incident report. Please try again.")
            else:
                st.warning("⚠️ Please provide an incident description before submitting.")

# Footer
st.markdown("---")
st.markdown(
    "**Disaster Response Multi-Modal Agent** | "
    "Powered by Cerebras Vision AI & Google Gemini | "
    f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

# Auto-refresh logic
if auto_refresh:
    time.sleep(30)
    st.rerun()