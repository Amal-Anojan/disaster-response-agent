import folium
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
from typing import Dict, Any, List, Optional
import random
from datetime import datetime, timedelta

class IncidentMap:
    def __init__(self):
        """Initialize incident map component"""
        self.default_location = [37.7749, -122.4194]  # San Francisco
        self.zoom_start = 12
        
        # Map tile options
        self.tile_options = {
            'OpenStreetMap': 'OpenStreetMap',
            'Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            'Terrain': 'Stamen Terrain'
        }
        
        # Incident type colors and icons
        self.incident_styles = {
            'fire': {
                'color': 'red',
                'icon': 'fire',
                'prefix': 'fa'
            },
            'flood': {
                'color': 'blue', 
                'icon': 'tint',
                'prefix': 'fa'
            },
            'earthquake': {
                'color': 'orange',
                'icon': 'home',
                'prefix': 'fa'
            },
            'accident': {
                'color': 'purple',
                'icon': 'car',
                'prefix': 'fa'
            },
            'medical': {
                'color': 'green',
                'icon': 'plus-square',
                'prefix': 'fa'
            },
            'hazmat': {
                'color': 'black',
                'icon': 'exclamation-triangle',
                'prefix': 'fa'
            },
            'storm': {
                'color': 'gray',
                'icon': 'cloud',
                'prefix': 'fa'
            },
            'unknown': {
                'color': 'darkblue',
                'icon': 'question',
                'prefix': 'fa'
            }
        }
        
        # Urgency level styles
        self.urgency_styles = {
            'CRITICAL': {'color': 'red', 'fill_opacity': 0.8},
            'HIGH': {'color': 'orange', 'fill_opacity': 0.6},
            'MEDIUM': {'color': 'yellow', 'fill_opacity': 0.4},
            'LOW': {'color': 'green', 'fill_opacity': 0.3}
        }
    
    def create_map(self, 
                   incidents: List[Dict[str, Any]], 
                   center_location: Optional[List[float]] = None,
                   tile_layer: str = 'OpenStreetMap',
                   show_heatmap: bool = False) -> folium.Map:
        """
        Create interactive incident map
        
        Args:
            incidents: List of incident data
            center_location: Map center coordinates [lat, lng]
            tile_layer: Map tile layer to use
            show_heatmap: Whether to show incident density heatmap
            
        Returns:
            Folium map object
        """
        # Determine map center
        if center_location:
            map_center = center_location
        elif incidents:
            # Center on incidents
            lats = [inc.get('location', {}).get('lat', self.default_location[0]) for inc in incidents]
            lngs = [inc.get('location', {}).get('lng', self.default_location[1]) for inc in incidents]
            map_center = [sum(lats) / len(lats), sum(lngs) / len(lngs)]
        else:
            map_center = self.default_location
        
        # Create base map
        if tile_layer in self.tile_options:
            if tile_layer == 'OpenStreetMap':
                m = folium.Map(
                    location=map_center,
                    zoom_start=self.zoom_start,
                    tiles=self.tile_options[tile_layer]
                )
            else:
                m = folium.Map(
                    location=map_center,
                    zoom_start=self.zoom_start,
                    tiles=self.tile_options[tile_layer],
                    attr='Incident Map'
                )
        else:
            m = folium.Map(location=map_center, zoom_start=self.zoom_start)
        
        # Add incident markers
        for incident in incidents:
            self._add_incident_marker(m, incident)
        
        # Add heatmap layer if requested
        if show_heatmap and incidents:
            self._add_heatmap_layer(m, incidents)
        
        # Add legend
        self._add_legend(m)
        
        # Add measurement tool
        self._add_measurement_tool(m)
        
        return m
    
    def _add_incident_marker(self, map_obj: folium.Map, incident: Dict[str, Any]):
        """Add incident marker to map"""
        location = incident.get('location', {})
        if not location or 'lat' not in location or 'lng' not in location:
            return
        
        lat, lng = location['lat'], location['lng']
        
        # Get incident details
        incident_id = incident.get('incident_id', 'Unknown')
        disaster_type = incident.get('disaster_type', 'unknown').lower()
        urgency = incident.get('urgency', 'MEDIUM')
        severity = incident.get('severity', 5)
        status = incident.get('status', 'Unknown')
        timestamp = incident.get('timestamp', datetime.now().isoformat())
        text_content = incident.get('text_content', 'No description available')
        estimated_response_time = incident.get('estimated_response_time', 'Unknown')
        
        # Get style based on incident type
        style = self.incident_styles.get(disaster_type, self.incident_styles['unknown'])
        
        # Create popup content
        popup_html = f"""
        <div style="width: 300px;">
            <h4 style="color: {style['color']}; margin-bottom: 10px;">
                🚨 {disaster_type.title()} Emergency
            </h4>
            <p><strong>ID:</strong> {incident_id}</p>
            <p><strong>Severity:</strong> {severity}/10</p>
            <p><strong>Urgency:</strong> 
                <span style="background-color: {self.urgency_styles.get(urgency, {}).get('color', 'gray')}; 
                             color: white; padding: 2px 6px; border-radius: 3px;">
                    {urgency}
                </span>
            </p>
            <p><strong>Status:</strong> {status}</p>
            <p><strong>Response Time:</strong> {estimated_response_time} min</p>
            <p><strong>Time:</strong> {timestamp[:19].replace('T', ' ')}</p>
            <p><strong>Description:</strong><br>{text_content[:150]}{'...' if len(text_content) > 150 else ''}</p>
            
            <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #ccc;">
                <small><strong>Location:</strong> {lat:.4f}, {lng:.4f}</small>
            </div>
        </div>
        """
        
        # Create marker
        marker = folium.Marker(
            location=[lat, lng],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{disaster_type.title()} - {urgency} Priority",
            icon=folium.Icon(
                color=style['color'],
                icon=style['icon'],
                prefix=style['prefix']
            )
        )
        
        # Add marker to map
        marker.add_to(map_obj)
        
        # Add urgency circle around marker
        urgency_style = self.urgency_styles.get(urgency, {'color': 'blue', 'fill_opacity': 0.3})
        radius = max(100, severity * 50)  # Radius based on severity
        
        folium.Circle(
            location=[lat, lng],
            radius=radius,
            color=urgency_style['color'],
            fillColor=urgency_style['color'],
            fillOpacity=urgency_style['fill_opacity'],
            weight=2,
            opacity=0.8
        ).add_to(map_obj)
    
    def _add_heatmap_layer(self, map_obj: folium.Map, incidents: List[Dict[str, Any]]):
        """Add incident density heatmap"""
        try:
            from folium.plugins import HeatMap
            
            # Prepare heatmap data
            heat_data = []
            for incident in incidents:
                location = incident.get('location', {})
                if 'lat' in location and 'lng' in location:
                    lat, lng = location['lat'], location['lng']
                    # Weight by severity and urgency
                    severity = incident.get('severity', 5)
                    urgency = incident.get('urgency', 'MEDIUM')
                    urgency_weight = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(urgency, 2)
                    weight = severity * urgency_weight
                    heat_data.append([lat, lng, weight])
            
            if heat_data:
                # Add heatmap
                HeatMap(
                    heat_data,
                    min_opacity=0.2,
                    max_zoom=18,
                    radius=25,
                    blur=15,
                    gradient={'0.0': 'blue', '0.4': 'lime', '0.6': 'orange', '1.0': 'red'}
                ).add_to(map_obj)
                
        except ImportError:
            st.warning("HeatMap plugin not available. Install with: pip install folium[extras]")
    
    def _add_legend(self, map_obj: folium.Map):
        """Add legend to map"""
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);">
            <p><strong>Emergency Incidents</strong></p>
            <p><i class="fa fa-fire" style="color:red"></i> Fire</p>
            <p><i class="fa fa-tint" style="color:blue"></i> Flood</p>
            <p><i class="fa fa-home" style="color:orange"></i> Earthquake</p>
            <p><i class="fa fa-car" style="color:purple"></i> Accident</p>
            <p><i class="fa fa-plus-square" style="color:green"></i> Medical</p>
            <p><i class="fa fa-exclamation-triangle" style="color:black"></i> Hazmat</p>
            <br>
            <p><strong>Urgency Levels</strong></p>
            <p><span style="background-color:red; color:white; padding:2px 6px; border-radius:3px;">CRITICAL</span></p>
            <p><span style="background-color:orange; color:white; padding:2px 6px; border-radius:3px;">HIGH</span></p>
            <p><span style="background-color:yellow; color:black; padding:2px 6px; border-radius:3px;">MEDIUM</span></p>
            <p><span style="background-color:green; color:white; padding:2px 6px; border-radius:3px;">LOW</span></p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_measurement_tool(self, map_obj: folium.Map):
        """Add measurement tool for distance calculation"""
        try:
            from folium.plugins import MeasureControl
            
            # Add measurement control
            measure_control = MeasureControl(
                primary_length_unit='kilometers',
                secondary_length_unit='miles',
                primary_area_unit='sqkilometers',
                secondary_area_unit='acres'
            )
            map_obj.add_child(measure_control)
            
        except ImportError:
            pass  # Measurement tool not critical
    
    def render_map(self, 
                   incidents: List[Dict[str, Any]], 
                   map_height: int = 500,
                   map_key: str = "incident_map") -> Dict[str, Any]:
        """
        Render map in Streamlit and return interaction data
        
        Args:
            incidents: List of incident data
            map_height: Map height in pixels
            map_key: Unique key for Streamlit component
            
        Returns:
            Map interaction data
        """
        # Create map
        incident_map = self.create_map(incidents)
        
        # Render in Streamlit
        map_data = st_folium(
            incident_map,
            width=700,
            height=map_height,
            returned_objects=["last_object_clicked", "last_clicked"],
            key=map_key
        )
        
        return map_data
    
    def get_sample_incidents(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate sample incidents for testing"""
        disaster_types = ['fire', 'flood', 'earthquake', 'accident', 'medical', 'hazmat']
        urgency_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        statuses = ['PROCESSING', 'DISPATCHED', 'ON_SCENE', 'RESOLVED']
        
        # Base locations around San Francisco
        base_locations = [
            [37.7749, -122.4194],  # San Francisco
            [37.7849, -122.4094],  # North
            [37.7649, -122.4294],  # South
            [37.7749, -122.4094],  # East
            [37.7749, -122.4294],  # West
        ]
        
        incidents = []
        for i in range(count):
            base_loc = random.choice(base_locations)
            
            # Add some random offset
            lat_offset = random.uniform(-0.02, 0.02)
            lng_offset = random.uniform(-0.02, 0.02)
            
            disaster_type = random.choice(disaster_types)
            urgency = random.choice(urgency_levels)
            severity = random.randint(3, 10)
            
            # Generate realistic descriptions
            descriptions = {
                'fire': f"Building fire reported at {random.choice(['apartment complex', 'office building', 'residential home', 'warehouse'])}",
                'flood': f"Flooding due to {random.choice(['broken water main', 'heavy rainfall', 'dam overflow', 'storm surge'])}",
                'earthquake': f"Structural damage from {severity}.{random.randint(0,9)} magnitude earthquake",
                'accident': f"Multi-vehicle accident involving {random.randint(2,5)} vehicles",
                'medical': f"Mass casualty incident with {random.randint(3,15)} patients",
                'hazmat': f"Chemical spill at {random.choice(['industrial facility', 'transportation route', 'storage facility'])}"
            }
            
            incident = {
                'incident_id': f"INC_{datetime.now().strftime('%Y%m%d')}_{i+1:03d}",
                'disaster_type': disaster_type,
                'urgency': urgency,
                'severity': severity,
                'status': random.choice(statuses),
                'location': {
                    'lat': base_loc[0] + lat_offset,
                    'lng': base_loc[1] + lng_offset
                },
                'text_content': descriptions.get(disaster_type, "Emergency situation reported"),
                'estimated_response_time': random.randint(5, 30),
                'timestamp': (datetime.now() - timedelta(minutes=random.randint(0, 180))).isoformat()
            }
            
            incidents.append(incident)
        
        return incidents
    
    def filter_incidents_by_bounds(self, 
                                 incidents: List[Dict[str, Any]], 
                                 bounds: Dict[str, float]) -> List[Dict[str, Any]]:
        """Filter incidents by map bounds"""
        filtered = []
        
        for incident in incidents:
            location = incident.get('location', {})
            if 'lat' in location and 'lng' in location:
                lat, lng = location['lat'], location['lng']
                
                if (bounds['south'] <= lat <= bounds['north'] and 
                    bounds['west'] <= lng <= bounds['east']):
                    filtered.append(incident)
        
        return filtered
    
    def calculate_incident_statistics(self, incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for incidents on map"""
        if not incidents:
            return {
                'total_incidents': 0,
                'by_type': {},
                'by_urgency': {},
                'by_status': {},
                'average_severity': 0,
                'most_common_type': None
            }
        
        # Count by type
        by_type = {}
        by_urgency = {}
        by_status = {}
        severities = []
        
        for incident in incidents:
            disaster_type = incident.get('disaster_type', 'unknown')
            urgency = incident.get('urgency', 'MEDIUM')
            status = incident.get('status', 'Unknown')
            severity = incident.get('severity', 5)
            
            by_type[disaster_type] = by_type.get(disaster_type, 0) + 1
            by_urgency[urgency] = by_urgency.get(urgency, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1
            severities.append(severity)
        
        return {
            'total_incidents': len(incidents),
            'by_type': by_type,
            'by_urgency': by_urgency,
            'by_status': by_status,
            'average_severity': sum(severities) / len(severities) if severities else 0,
            'most_common_type': max(by_type.items(), key=lambda x: x[1])[0] if by_type else None
        }

# Usage example and testing
def test_incident_map():
    """Test the incident map component"""
    import streamlit as st
    
    st.title("Incident Map Test")
    
    # Create map instance
    incident_map = IncidentMap()
    
    # Generate sample data
    sample_incidents = incident_map.get_sample_incidents(15)
    
    # Display statistics
    stats = incident_map.calculate_incident_statistics(sample_incidents)
    st.write("**Incident Statistics:**")
    st.json(stats)
    
    # Render map
    st.write("**Interactive Incident Map:**")
    map_data = incident_map.render_map(sample_incidents, map_height=600)
    
    # Show clicked incident
    if map_data['last_object_clicked']:
        st.write("**Last Clicked:**")
        st.json(map_data['last_object_clicked'])

if __name__ == "__main__":
    test_incident_map()