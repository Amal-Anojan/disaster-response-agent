import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

class PriorityQueue:
    def __init__(self):
        """Initialize priority queue component"""
        self.priority_weights = {
            'CRITICAL': 4,
            'HIGH': 3,
            'MEDIUM': 2,
            'LOW': 1
        }
        
        self.severity_thresholds = {
            (9, 10): 'Catastrophic',
            (8, 9): 'Severe',
            (6, 8): 'Major',
            (4, 6): 'Moderate',
            (2, 4): 'Minor',
            (0, 2): 'Minimal'
        }
        
        self.status_styles = {
            'PROCESSING': {'color': '#FFA500', 'icon': '⚙️'},
            'DISPATCHED': {'color': '#1E90FF', 'icon': '🚗'},
            'ON_SCENE': {'color': '#FF4500', 'icon': '🎯'},
            'RESOLVED': {'color': '#32CD32', 'icon': '✅'},
            'CANCELLED': {'color': '#808080', 'icon': '❌'}
        }
    
    def calculate_priority_score(self, incident: Dict[str, Any]) -> float:
        """
        Calculate priority score for incident
        
        Args:
            incident: Incident data dictionary
            
        Returns:
            Priority score (higher = more urgent)
        """
        # Base components
        severity = incident.get('severity', 5.0)
        urgency = incident.get('urgency', 'MEDIUM')
        affected_population = incident.get('estimated_affected_population', 0)
        
        # Urgency weight
        urgency_weight = self.priority_weights.get(urgency, 2)
        
        # Base priority calculation
        base_score = severity * urgency_weight
        
        # Population multiplier
        if affected_population > 500:
            population_multiplier = 1.5
        elif affected_population > 100:
            population_multiplier = 1.2
        else:
            population_multiplier = 1.0
        
        # Time factor (more recent = higher priority)
        timestamp = incident.get('timestamp')
        if timestamp:
            try:
                incident_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_diff = (datetime.now() - incident_time.replace(tzinfo=None)).total_seconds() / 3600
                
                # Boost priority for very recent incidents
                if time_diff < 0.5:  # Less than 30 minutes
                    time_factor = 1.3
                elif time_diff < 2:   # Less than 2 hours
                    time_factor = 1.1
                else:
                    time_factor = 1.0
            except:
                time_factor = 1.0
        else:
            time_factor = 1.0
        
        # Infrastructure impact
        infrastructure_impact = 1.0
        affected_infrastructure = incident.get('affected_infrastructure', [])
        critical_infrastructure = ['hospital', 'power_plant', 'emergency_services', 'communications']
        
        if any(crit in str(affected_infrastructure).lower() for crit in critical_infrastructure):
            infrastructure_impact = 1.4
        
        # Calculate final score
        final_score = base_score * population_multiplier * time_factor * infrastructure_impact
        
        return round(final_score, 2)
    
    def sort_incidents_by_priority(self, incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort incidents by priority score"""
        incidents_with_priority = []
        
        for incident in incidents:
            incident_copy = incident.copy()
            incident_copy['priority_score'] = self.calculate_priority_score(incident)
            incidents_with_priority.append(incident_copy)
        
        # Sort by priority score (descending) and then by timestamp (most recent first)
        return sorted(
            incidents_with_priority,
            key=lambda x: (x['priority_score'], x.get('timestamp', '')),
            reverse=True
        )
    
    def get_severity_label(self, severity: float) -> str:
        """Get severity label from numeric value"""
        for (min_val, max_val), label in self.severity_thresholds.items():
            if min_val <= severity < max_val:
                return label
        return 'Unknown'
    
    def render_priority_queue(self, 
                            incidents: List[Dict[str, Any]], 
                            max_items: int = 20,
                            show_filters: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Render priority queue in Streamlit
        
        Args:
            incidents: List of incident data
            max_items: Maximum items to display
            show_filters: Whether to show filter controls
            
        Returns:
            Tuple of (filtered_incidents, filter_settings)
        """
        # Initialize filter settings
        filter_settings = {}
        
        if show_filters:
            st.subheader("🔍 Priority Queue Filters")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                urgency_filter = st.multiselect(
                    "Urgency Level",
                    options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
                    default=['CRITICAL', 'HIGH'],
                    key="priority_urgency_filter"
                )
                filter_settings['urgency'] = urgency_filter
            
            with col2:
                status_filter = st.multiselect(
                    "Status",
                    options=['PROCESSING', 'DISPATCHED', 'ON_SCENE', 'RESOLVED'],
                    default=['PROCESSING', 'DISPATCHED', 'ON_SCENE'],
                    key="priority_status_filter"
                )
                filter_settings['status'] = status_filter
            
            with col3:
                min_severity = st.slider(
                    "Minimum Severity",
                    min_value=1.0,
                    max_value=10.0,
                    value=4.0,
                    step=0.5,
                    key="priority_min_severity"
                )
                filter_settings['min_severity'] = min_severity
            
            with col4:
                disaster_types = list(set([inc.get('disaster_type', 'unknown') for inc in incidents]))
                type_filter = st.multiselect(
                    "Disaster Type",
                    options=disaster_types,
                    default=disaster_types,
                    key="priority_type_filter"
                )
                filter_settings['disaster_types'] = type_filter
        
        # Apply filters
        filtered_incidents = self._apply_filters(incidents, filter_settings if show_filters else {})
        
        # Sort by priority
        sorted_incidents = self.sort_incidents_by_priority(filtered_incidents)
        
        # Display queue header
        st.subheader(f"📋 Priority Queue ({len(sorted_incidents)} incidents)")
        
        if not sorted_incidents:
            st.info("No incidents match the current filters.")
            return [], filter_settings
        
        # Display priority summary
        self._render_priority_summary(sorted_incidents[:max_items])
        
        # Display incident cards
        displayed_incidents = self._render_incident_cards(sorted_incidents[:max_items])
        
        return displayed_incidents, filter_settings
    
    def _apply_filters(self, incidents: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to incident list"""
        if not filters:
            return incidents
        
        filtered = incidents.copy()
        
        # Urgency filter
        if 'urgency' in filters and filters['urgency']:
            filtered = [inc for inc in filtered if inc.get('urgency', 'MEDIUM') in filters['urgency']]
        
        # Status filter
        if 'status' in filters and filters['status']:
            filtered = [inc for inc in filtered if inc.get('status', 'PROCESSING') in filters['status']]
        
        # Severity filter
        if 'min_severity' in filters:
            filtered = [inc for inc in filtered if inc.get('severity', 5.0) >= filters['min_severity']]
        
        # Disaster type filter
        if 'disaster_types' in filters and filters['disaster_types']:
            filtered = [inc for inc in filtered if inc.get('disaster_type', 'unknown') in filters['disaster_types']]
        
        return filtered
    
    def _render_priority_summary(self, incidents: List[Dict[str, Any]]):
        """Render priority queue summary statistics"""
        if not incidents:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Top priority incident
        top_incident = incidents[0]
        with col1:
            st.metric(
                "Highest Priority",
                f"{top_incident['priority_score']:.1f}",
                f"{top_incident.get('disaster_type', 'Unknown').title()}"
            )
        
        # Critical incidents
        critical_count = len([inc for inc in incidents if inc.get('urgency') == 'CRITICAL'])
        with col2:
            st.metric(
                "Critical Incidents",
                critical_count,
                f"{critical_count/len(incidents)*100:.0f}% of queue"
            )
        
        # Average severity
        avg_severity = sum(inc.get('severity', 5) for inc in incidents) / len(incidents)
        with col3:
            st.metric(
                "Average Severity",
                f"{avg_severity:.1f}/10",
                self.get_severity_label(avg_severity)
            )
        
        # Response time estimate
        avg_response_time = sum(inc.get('estimated_response_time', 15) for inc in incidents) / len(incidents)
        with col4:
            st.metric(
                "Avg Response Time",
                f"{avg_response_time:.0f} min",
                "Estimated"
            )
    
    def _render_incident_cards(self, incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Render individual incident cards"""
        displayed_incidents = []
        
        for i, incident in enumerate(incidents):
            with st.container():
                # Create card styling based on priority
                priority_score = incident.get('priority_score', 0)
                urgency = incident.get('urgency', 'MEDIUM')
                status = incident.get('status', 'PROCESSING')
                
                # Card header with priority and urgency
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**#{i+1} - {incident.get('incident_id', 'Unknown')}**")
                    disaster_type = incident.get('disaster_type', 'unknown').title()
                    st.markdown(f"🚨 **{disaster_type}** Emergency")
                
                with col2:
                    # Priority score with color coding
                    if priority_score >= 20:
                        priority_color = "🔴"
                    elif priority_score >= 15:
                        priority_color = "🟠"
                    elif priority_score >= 10:
                        priority_color = "🟡"
                    else:
                        priority_color = "🟢"
                    
                    st.markdown(f"**Priority:** {priority_color} {priority_score:.1f}")
                    
                    # Urgency badge
                    urgency_color = {
                        'CRITICAL': '🔴',
                        'HIGH': '🟠', 
                        'MEDIUM': '🟡',
                        'LOW': '🟢'
                    }.get(urgency, '⚪')
                    st.markdown(f"**Urgency:** {urgency_color} {urgency}")
                
                with col3:
                    # Status
                    status_info = self.status_styles.get(status, {'color': '#808080', 'icon': '❓'})
                    st.markdown(f"**Status:** {status_info['icon']} {status}")
                    
                    # Severity
                    severity = incident.get('severity', 5)
                    severity_label = self.get_severity_label(severity)
                    st.markdown(f"**Severity:** {severity:.1f}/10 ({severity_label})")
                
                # Incident details
                col_details1, col_details2 = st.columns([2, 1])
                
                with col_details1:
                    # Description
                    description = incident.get('text_content', 'No description available')
                    st.markdown(f"**Description:** {description[:200]}{'...' if len(description) > 200 else ''}")
                    
                    # Location
                    location = incident.get('location', {})
                    if location and 'lat' in location and 'lng' in location:
                        st.markdown(f"**Location:** {location['lat']:.4f}, {location['lng']:.4f}")
                    
                    # Affected infrastructure
                    affected_infra = incident.get('affected_infrastructure', [])
                    if affected_infra:
                        infra_str = ', '.join(affected_infra[:3])
                        if len(affected_infra) > 3:
                            infra_str += f" and {len(affected_infra) - 3} more"
                        st.markdown(f"**Affected Infrastructure:** {infra_str}")
                
                with col_details2:
                    # Timeline information
                    timestamp = incident.get('timestamp')
                    if timestamp:
                        try:
                            incident_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_ago = datetime.now() - incident_time.replace(tzinfo=None)
                            if time_ago.total_seconds() < 3600:
                                time_str = f"{int(time_ago.total_seconds() // 60)} min ago"
                            else:
                                time_str = f"{int(time_ago.total_seconds() // 3600)} hr ago"
                            st.markdown(f"**Reported:** {time_str}")
                        except:
                            st.markdown(f"**Reported:** Unknown")
                    
                    # Response metrics
                    response_time = incident.get('estimated_response_time', 'Unknown')
                    st.markdown(f"**Response ETA:** {response_time} min")
                    
                    # Affected population
                    affected_pop = incident.get('estimated_affected_population', 0)
                    if affected_pop > 0:
                        st.markdown(f"**Affected People:** ~{affected_pop}")
                    
                    # Action buttons
                    button_col1, button_col2 = st.columns(2)
                    with button_col1:
                        if st.button("📋 Details", key=f"details_{incident.get('incident_id', i)}"):
                            st.session_state[f"show_details_{incident.get('incident_id', i)}"] = True
                    
                    with button_col2:
                        if st.button("🚀 Dispatch", key=f"dispatch_{incident.get('incident_id', i)}"):
                            st.success(f"Dispatching resources for {incident.get('incident_id', 'incident')}")
                
                # Show detailed view if requested
                if st.session_state.get(f"show_details_{incident.get('incident_id', i)}", False):
                    with st.expander("📋 Detailed Information", expanded=True):
                        self._render_incident_details(incident)
                
                # Add separator
                st.markdown("---")
                
                displayed_incidents.append(incident)
        
        return displayed_incidents
    
    def _render_incident_details(self, incident: Dict[str, Any]):
        """Render detailed incident information"""
        # Analysis results
        if incident.get('vision_analysis'):
            st.markdown("**🔍 Vision Analysis:**")
            vision_analysis = incident['vision_analysis']
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• Confidence: {vision_analysis.get('confidence', 0.5):.1%}")
                st.write(f"• Processing Time: {vision_analysis.get('processing_time', 'Unknown')}")
            with col2:
                hazards = vision_analysis.get('hazards', [])
                if hazards:
                    st.write(f"• Hazards: {', '.join(hazards[:3])}")
        
        # Action plan
        if incident.get('action_plan'):
            st.markdown("**📋 Action Plan:**")
            action_plan = incident['action_plan']
            
            immediate_actions = action_plan.get('immediate_actions', [])
            if immediate_actions:
                st.markdown("*Immediate Actions:*")
                for action in immediate_actions[:5]:
                    st.write(f"• {action}")
            
            # Resource requirements
            resources = action_plan.get('resource_requirements', {})
            if resources:
                st.markdown("*Resource Requirements:*")
                personnel = resources.get('personnel', {})
                if personnel:
                    for role, count in personnel.items():
                        st.write(f"• {role.replace('_', ' ').title()}: {count}")
        
        # Timeline
        if incident.get('timeline'):
            st.markdown("**⏱️ Timeline:**")
            timeline = incident['timeline']
            for phase, actions in timeline.items():
                st.write(f"**{phase.replace('_', '-')}:**")
                for action in actions[:3]:
                    st.write(f"  • {action}")
    
    def render_priority_charts(self, incidents: List[Dict[str, Any]]):
        """Render priority analysis charts"""
        if not incidents:
            st.info("No incidents to analyze")
            return
        
        # Add priority scores
        incidents_with_priority = self.sort_incidents_by_priority(incidents)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Priority distribution
            priority_scores = [inc['priority_score'] for inc in incidents_with_priority]
            fig_priority = px.histogram(
                x=priority_scores,
                nbins=20,
                title="Priority Score Distribution",
                labels={'x': 'Priority Score', 'y': 'Count'},
                color_discrete_sequence=['#ff6b6b']
            )
            fig_priority.update_layout(height=350)
            st.plotly_chart(fig_priority, use_container_width=True)
        
        with col2:
            # Urgency vs Severity scatter
            urgency_numeric = [self.priority_weights.get(inc.get('urgency', 'MEDIUM'), 2) 
                             for inc in incidents_with_priority]
            severity_values = [inc.get('severity', 5) for inc in incidents_with_priority]
            
            fig_scatter = px.scatter(
                x=severity_values,
                y=urgency_numeric,
                title="Severity vs Urgency",
                labels={'x': 'Severity (1-10)', 'y': 'Urgency Level'},
                color=priority_scores,
                color_continuous_scale='Reds'
            )
            fig_scatter.update_layout(height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def export_priority_queue(self, incidents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Export priority queue as DataFrame"""
        sorted_incidents = self.sort_incidents_by_priority(incidents)
        
        export_data = []
        for i, incident in enumerate(sorted_incidents):
            export_data.append({
                'Rank': i + 1,
                'Incident_ID': incident.get('incident_id', 'Unknown'),
                'Priority_Score': incident.get('priority_score', 0),
                'Disaster_Type': incident.get('disaster_type', 'unknown'),
                'Urgency': incident.get('urgency', 'MEDIUM'),
                'Severity': incident.get('severity', 5),
                'Status': incident.get('status', 'PROCESSING'),
                'Affected_Population': incident.get('estimated_affected_population', 0),
                'Response_Time_Min': incident.get('estimated_response_time', 0),
                'Description': incident.get('text_content', '')[:100],
                'Timestamp': incident.get('timestamp', '')
            })
        
        return pd.DataFrame(export_data)

# Usage example and testing
def test_priority_queue():
    """Test the priority queue component"""
    import random
    from datetime import datetime, timedelta
    
    st.title("Priority Queue Test")
    
    # Generate sample incidents
    sample_incidents = []
    disaster_types = ['fire', 'flood', 'earthquake', 'accident', 'medical']
    urgency_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    statuses = ['PROCESSING', 'DISPATCHED', 'ON_SCENE']
    
    for i in range(25):
        incident = {
            'incident_id': f"INC_{datetime.now().strftime('%Y%m%d')}_{i+1:03d}",
            'disaster_type': random.choice(disaster_types),
            'urgency': random.choice(urgency_levels),
            'severity': random.uniform(2.0, 10.0),
            'status': random.choice(statuses),
            'estimated_affected_population': random.randint(10, 1000),
            'estimated_response_time': random.randint(5, 30),
            'text_content': f"Sample emergency incident #{i+1} with various details and description",
            'affected_infrastructure': random.sample(['roads', 'buildings', 'power_lines', 'hospitals'], 2),
            'timestamp': (datetime.now() - timedelta(minutes=random.randint(0, 300))).isoformat(),
            'location': {'lat': 37.7749 + random.uniform(-0.1, 0.1), 'lng': -122.4194 + random.uniform(-0.1, 0.1)}
        }
        sample_incidents.append(incident)
    
    # Create priority queue
    priority_queue = PriorityQueue()
    
    # Render priority queue
    displayed_incidents, filters = priority_queue.render_priority_queue(sample_incidents, max_items=15)
    
    # Render charts
    st.subheader("📊 Priority Analysis")
    priority_queue.render_priority_charts(displayed_incidents)
    
    # Export functionality
    if st.button("📤 Export Priority Queue"):
        df = priority_queue.export_priority_queue(displayed_incidents)
        st.dataframe(df)

if __name__ == "__main__":
    test_priority_queue()