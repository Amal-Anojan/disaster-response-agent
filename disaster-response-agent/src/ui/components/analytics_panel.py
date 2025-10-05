import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np

class AnalyticsPanel:
    def __init__(self):
        """Initialize analytics panel component"""
        self.color_palette = {
            'primary': '#FF4B4B',
            'secondary': '#1f77b4',
            'success': '#32CD32',
            'warning': '#FFA500',
            'danger': '#FF6347',
            'info': '#17a2b8'
        }
        
        self.disaster_colors = {
            'fire': '#FF4500',
            'flood': '#4169E1',
            'earthquake': '#FF8C00',
            'accident': '#9932CC',
            'medical': '#32CD32',
            'hazmat': '#2F4F4F',
            'storm': '#708090'
        }
    
    def render_analytics_dashboard(self, incidents: List[Dict[str, Any]]):
        """Render complete analytics dashboard"""
        if not incidents:
            st.info("No incident data available for analytics")
            return
        
        # Convert to DataFrame for easier analysis
        df = self._incidents_to_dataframe(incidents)
        
        # Dashboard header
        st.header("📊 Emergency Response Analytics")
        
        # Key metrics row
        self._render_key_metrics(df)
        
        # Main analytics tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Trends", "🗺️ Geographic", "⚡ Performance", "🎯 Predictions", "📋 Reports"
        ])
        
        with tab1:
            self._render_trend_analysis(df)
        
        with tab2:
            self._render_geographic_analysis(df)
        
        with tab3:
            self._render_performance_metrics(df)
        
        with tab4:
            self._render_predictive_analysis(df)
        
        with tab5:
            self._render_reports(df)
    
    def _incidents_to_dataframe(self, incidents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert incident list to pandas DataFrame"""
        data = []
        
        for incident in incidents:
            # Parse timestamp
            timestamp = incident.get('timestamp', datetime.now().isoformat())
            try:
                parsed_time = pd.to_datetime(timestamp)
            except:
                parsed_time = pd.to_datetime(datetime.now())
            
            # Extract location
            location = incident.get('location', {})
            lat = location.get('lat', 37.7749)
            lng = location.get('lng', -122.4194)
            
            # Build row data
            row = {
                'incident_id': incident.get('incident_id', 'Unknown'),
                'timestamp': parsed_time,
                'disaster_type': incident.get('disaster_type', 'unknown'),
                'severity': incident.get('severity', 5.0),
                'urgency': incident.get('urgency', 'MEDIUM'),
                'status': incident.get('status', 'PROCESSING'),
                'affected_population': incident.get('estimated_affected_population', 0),
                'response_time': incident.get('estimated_response_time', 15),
                'lat': lat,
                'lng': lng,
                'description_length': len(incident.get('text_content', '')),
                'infrastructure_count': len(incident.get('affected_infrastructure', [])),
                'hour': parsed_time.hour,
                'day_of_week': parsed_time.dayofweek,
                'month': parsed_time.month
            }
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _render_key_metrics(self, df: pd.DataFrame):
        """Render key performance metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Total incidents
        total_incidents = len(df)
        recent_incidents = len(df[df['timestamp'] > (datetime.now() - timedelta(hours=24))])
        
        with col1:
            st.metric(
                "Total Incidents",
                total_incidents,
                delta=recent_incidents,
                delta_color="inverse"
            )
        
        # Average severity
        avg_severity = df['severity'].mean()
        high_severity_count = len(df[df['severity'] >= 7])
        
        with col2:
            st.metric(
                "Avg Severity",
                f"{avg_severity:.1f}/10",
                delta=f"{high_severity_count} high severity",
                delta_color="off"
            )
        
        # Critical incidents
        critical_count = len(df[df['urgency'] == 'CRITICAL'])
        critical_percentage = (critical_count / total_incidents * 100) if total_incidents > 0 else 0
        
        with col3:
            st.metric(
                "Critical Incidents",
                critical_count,
                delta=f"{critical_percentage:.1f}% of total",
                delta_color="inverse"
            )
        
        # Average response time
        avg_response_time = df['response_time'].mean()
        fast_responses = len(df[df['response_time'] <= 10])
        
        with col4:
            st.metric(
                "Avg Response Time",
                f"{avg_response_time:.0f} min",
                delta=f"{fast_responses} under 10min",
                delta_color="normal"
            )
        
        # Affected population
        total_affected = df['affected_population'].sum()
        avg_affected = df['affected_population'].mean()
        
        with col5:
            st.metric(
                "Total Affected",
                f"{total_affected:,}",
                delta=f"Avg: {avg_affected:.0f}",
                delta_color="off"
            )
    
    def _render_trend_analysis(self, df: pd.DataFrame):
        """Render trend analysis charts"""
        st.subheader("📈 Incident Trends Over Time")
        
        # Time series controls
        col1, col2 = st.columns([3, 1])
        
        with col2:
            time_grouping = st.selectbox(
                "Group by",
                ["Hour", "Day", "Week", "Month"],
                index=1
            )
        
        # Incident trends over time
        col1, col2 = st.columns(2)
        
        with col1:
            # Incidents by time
            if time_grouping == "Hour":
                time_data = df.groupby(df['timestamp'].dt.floor('H')).size().reset_index()
                time_data.columns = ['time', 'count']
            elif time_grouping == "Day":
                time_data = df.groupby(df['timestamp'].dt.date).size().reset_index()
                time_data.columns = ['time', 'count']
            elif time_grouping == "Week":
                time_data = df.groupby(df['timestamp'].dt.to_period('W')).size().reset_index()
                time_data['time'] = time_data['timestamp'].astype(str)
                time_data.columns = ['temp', 'time', 'count']
                time_data = time_data[['time', 'count']]
            else:  # Month
                time_data = df.groupby(df['timestamp'].dt.to_period('M')).size().reset_index()
                time_data['time'] = time_data['timestamp'].astype(str)
                time_data.columns = ['temp', 'time', 'count']
                time_data = time_data[['time', 'count']]
            
            fig_timeline = px.line(
                time_data, 
                x='time', 
                y='count',
                title=f"Incidents by {time_grouping}",
                color_discrete_sequence=[self.color_palette['primary']]
            )
            fig_timeline.update_layout(height=350)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            # Severity trend
            severity_trend = df.groupby(df['timestamp'].dt.date)['severity'].mean().reset_index()
            severity_trend.columns = ['date', 'avg_severity']
            
            fig_severity = px.line(
                severity_trend,
                x='date',
                y='avg_severity',
                title="Average Severity Trend",
                color_discrete_sequence=[self.color_palette['danger']]
            )
            fig_severity.update_layout(height=350)
            fig_severity.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="High Severity Threshold")
            st.plotly_chart(fig_severity, use_container_width=True)
        
        # Disaster type distribution
        st.subheader("🔥 Incident Type Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of disaster types
            type_counts = df['disaster_type'].value_counts()
            
            fig_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Distribution by Disaster Type",
                color_discrete_map=self.disaster_colors
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Disaster type severity comparison
            severity_by_type = df.groupby('disaster_type')['severity'].agg(['mean', 'std', 'count']).reset_index()
            
            fig_box = px.box(
                df,
                x='disaster_type',
                y='severity',
                title="Severity Distribution by Type",
                color='disaster_type',
                color_discrete_map=self.disaster_colors
            )
            fig_box.update_layout(height=400, showlegend=False)
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
    
    def _render_geographic_analysis(self, df: pd.DataFrame):
        """Render geographic analysis"""
        st.subheader("🗺️ Geographic Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Incident density heatmap (simplified)
            st.markdown("**Incident Hotspots**")
            
            # Create bins for location clustering
            lat_bins = pd.cut(df['lat'], bins=10, labels=False)
            lng_bins = pd.cut(df['lng'], bins=10, labels=False)
            
            # Count incidents per grid cell
            location_counts = df.groupby([lat_bins, lng_bins]).size().reset_index()
            location_counts.columns = ['lat_bin', 'lng_bin', 'count']
            
            # Get actual coordinates for centers
            lat_centers = [(df['lat'].min() + (df['lat'].max() - df['lat'].min()) * (i + 0.5) / 10) for i in range(10)]
            lng_centers = [(df['lng'].min() + (df['lng'].max() - df['lng'].min()) * (i + 0.5) / 10) for i in range(10)]
            
            # Create heatmap data
            heatmap_data = []
            for _, row in location_counts.iterrows():
                if not pd.isna(row['lat_bin']) and not pd.isna(row['lng_bin']):
                    lat_idx = int(row['lat_bin'])
                    lng_idx = int(row['lng_bin'])
                    if 0 <= lat_idx < len(lat_centers) and 0 <= lng_idx < len(lng_centers):
                        heatmap_data.append([lat_centers[lat_idx], lng_centers[lng_idx], row['count']])
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data, columns=['lat', 'lng', 'count'])
                
                fig_geo = px.density_mapbox(
                    heatmap_df,
                    lat='lat',
                    lon='lng',
                    z='count',
                    radius=20,
                    center=dict(lat=df['lat'].mean(), lon=df['lng'].mean()),
                    zoom=10,
                    mapbox_style="open-street-map",
                    title="Incident Density"
                )
                fig_geo.update_layout(height=400)
                st.plotly_chart(fig_geo, use_container_width=True)
            else:
                st.info("Insufficient geographic data for heatmap")
        
        with col2:
            # Response time by location
            st.markdown("**Response Time Analysis**")
            
            # Calculate distance from center (proxy for response time analysis)
            center_lat, center_lng = df['lat'].mean(), df['lng'].mean()
            df['distance_from_center'] = np.sqrt(
                (df['lat'] - center_lat)**2 + (df['lng'] - center_lng)**2
            )
            
            fig_scatter = px.scatter(
                df,
                x='distance_from_center',
                y='response_time',
                color='disaster_type',
                size='severity',
                title="Response Time vs Distance",
                color_discrete_map=self.disaster_colors,
                labels={
                    'distance_from_center': 'Distance from Center',
                    'response_time': 'Response Time (min)'
                }
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Geographic statistics
        st.subheader("📍 Geographic Statistics")
        
        geo_col1, geo_col2, geo_col3, geo_col4 = st.columns(4)
        
        with geo_col1:
            # Most active area
            most_incidents_lat = df.loc[df.groupby('lat')['lat'].transform('count').idxmax(), 'lat']
            st.metric("Most Active Lat", f"{most_incidents_lat:.4f}")
        
        with geo_col2:
            most_incidents_lng = df.loc[df.groupby('lng')['lng'].transform('count').idxmax(), 'lng']
            st.metric("Most Active Lng", f"{most_incidents_lng:.4f}")
        
        with geo_col3:
            # Geographic spread
            lat_range = df['lat'].max() - df['lat'].min()
            st.metric("Latitude Range", f"{lat_range:.4f}°")
        
        with geo_col4:
            lng_range = df['lng'].max() - df['lng'].min()
            st.metric("Longitude Range", f"{lng_range:.4f}°")
    
    def _render_performance_metrics(self, df: pd.DataFrame):
        """Render performance analysis"""
        st.subheader("⚡ Response Performance Analysis")
        
        # Response time analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time distribution
            fig_hist = px.histogram(
                df,
                x='response_time',
                nbins=20,
                title="Response Time Distribution",
                color_discrete_sequence=[self.color_palette['info']]
            )
            fig_hist.add_vline(x=df['response_time'].mean(), line_dash="dash", line_color="red", 
                              annotation_text=f"Avg: {df['response_time'].mean():.1f} min")
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Response time by urgency
            fig_box_urgency = px.box(
                df,
                x='urgency',
                y='response_time',
                title="Response Time by Urgency",
                color='urgency',
                color_discrete_sequence=['green', 'yellow', 'orange', 'red']
            )
            fig_box_urgency.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_box_urgency, use_container_width=True)
        
        # Performance metrics table
        st.subheader("📊 Performance Summary")
        
        performance_metrics = []
        
        for disaster_type in df['disaster_type'].unique():
            type_data = df[df['disaster_type'] == disaster_type]
            
            metrics = {
                'Disaster Type': disaster_type.title(),
                'Total Incidents': len(type_data),
                'Avg Severity': f"{type_data['severity'].mean():.1f}",
                'Avg Response Time (min)': f"{type_data['response_time'].mean():.1f}",
                'Critical Count': len(type_data[type_data['urgency'] == 'CRITICAL']),
                'Total Affected': f"{type_data['affected_population'].sum():,}",
                'Success Rate': f"{len(type_data[type_data['status'] == 'RESOLVED']) / len(type_data) * 100:.1f}%"
            }
            
            performance_metrics.append(metrics)
        
        performance_df = pd.DataFrame(performance_metrics)
        st.dataframe(performance_df, use_container_width=True)
        
        # Efficiency trends
        st.subheader("📈 Efficiency Trends")
        
        # Calculate daily efficiency metrics
        daily_metrics = df.groupby(df['timestamp'].dt.date).agg({
            'response_time': 'mean',
            'severity': 'mean',
            'affected_population': 'sum'
        }).reset_index()
        
        # Create subplot for multiple metrics
        fig_efficiency = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Avg Response Time', 'Avg Severity', 'Total Affected Population', 'Incident Count'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        fig_efficiency.add_trace(
            go.Scatter(x=daily_metrics['timestamp'], y=daily_metrics['response_time'], 
                      name='Response Time', line=dict(color=self.color_palette['primary'])),
            row=1, col=1
        )
        
        fig_efficiency.add_trace(
            go.Scatter(x=daily_metrics['timestamp'], y=daily_metrics['severity'],
                      name='Severity', line=dict(color=self.color_palette['danger'])),
            row=1, col=2
        )
        
        fig_efficiency.add_trace(
            go.Scatter(x=daily_metrics['timestamp'], y=daily_metrics['affected_population'],
                      name='Affected Population', line=dict(color=self.color_palette['warning'])),
            row=2, col=1
        )
        
        # Daily incident count
        daily_counts = df.groupby(df['timestamp'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'count']
        
        fig_efficiency.add_trace(
            go.Scatter(x=daily_counts['date'], y=daily_counts['count'],
                      name='Incident Count', line=dict(color=self.color_palette['info'])),
            row=2, col=2
        )
        
        fig_efficiency.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    def _render_predictive_analysis(self, df: pd.DataFrame):
        """Render predictive analysis and forecasting"""
        st.subheader("🎯 Predictive Analysis & Forecasting")
        
        # Time-based patterns
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern
            hourly_pattern = df.groupby('hour').size().reset_index()
            hourly_pattern.columns = ['hour', 'count']
            
            fig_hourly = px.bar(
                hourly_pattern,
                x='hour',
                y='count',
                title="Incidents by Hour of Day",
                color='count',
                color_continuous_scale='Reds'
            )
            fig_hourly.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Peak hours analysis
            peak_hour = hourly_pattern.loc[hourly_pattern['count'].idxmax(), 'hour']
            st.info(f"Peak incident hour: {peak_hour}:00")
        
        with col2:
            # Day of week pattern
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_pattern = df.groupby('day_of_week').size().reset_index()
            daily_pattern.columns = ['day_of_week', 'count']
            daily_pattern['day_name'] = [day_names[i] for i in daily_pattern['day_of_week']]
            
            fig_daily = px.bar(
                daily_pattern,
                x='day_name',
                y='count',
                title="Incidents by Day of Week",
                color='count',
                color_continuous_scale='Blues'
            )
            fig_daily.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Peak day analysis
            peak_day_idx = daily_pattern.loc[daily_pattern['count'].idxmax(), 'day_of_week']
            st.info(f"Peak incident day: {day_names[peak_day_idx]}")
        
        # Risk prediction matrix
        st.subheader("⚠️ Risk Assessment Matrix")
        
        # Create risk score based on historical patterns
        risk_factors = df.groupby(['disaster_type', 'hour']).agg({
            'severity': 'mean',
            'affected_population': 'mean',
            'incident_id': 'count'
        }).reset_index()
        
        risk_factors['risk_score'] = (
            risk_factors['severity'] * 0.4 +
            (risk_factors['affected_population'] / 100) * 0.3 +
            risk_factors['incident_id'] * 0.3
        )
        
        # Create risk heatmap
        risk_pivot = risk_factors.pivot(index='disaster_type', columns='hour', values='risk_score')
        risk_pivot = risk_pivot.fillna(0)
        
        fig_heatmap = px.imshow(
            risk_pivot.values,
            x=[f"{h}:00" for h in risk_pivot.columns],
            y=risk_pivot.index,
            title="Risk Score Heatmap (Disaster Type vs Hour)",
            color_continuous_scale='Reds',
            aspect="auto"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Forecasting insights
        st.subheader("🔮 Forecasting Insights")
        
        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
        
        with forecast_col1:
            # Next high-risk period
            current_hour = datetime.now().hour
            next_hours = [(current_hour + i) % 24 for i in range(1, 25)]
            
            hourly_risk = df.groupby('hour')['severity'].mean()
            next_period_risks = [hourly_risk.get(h, 5.0) for h in next_hours[:6]]
            peak_next_hour = next_hours[np.argmax(next_period_risks)]
            
            st.metric(
                "Next High-Risk Hour",
                f"{peak_next_hour}:00",
                f"Risk Score: {max(next_period_risks):.1f}"
            )
        
        with forecast_col2:
            # Expected incidents today
            hour_avg = len(df) / max(len(df['timestamp'].dt.date.unique()), 1)
            remaining_hours = 24 - current_hour
            expected_remaining = hour_avg * remaining_hours / 24
            
            st.metric(
                "Expected Remaining Today",
                f"{expected_remaining:.0f}",
                "Based on historical avg"
            )
        
        with forecast_col3:
            # Most likely incident type
            type_probs = df['disaster_type'].value_counts(normalize=True)
            most_likely_type = type_probs.index[0]
            probability = type_probs.iloc[0]
            
            st.metric(
                "Most Likely Next Type",
                most_likely_type.title(),
                f"{probability:.1%} probability"
            )
    
    def _render_reports(self, df: pd.DataFrame):
        """Render report generation and export options"""
        st.subheader("📋 Reports & Export")
        
        # Report type selection
        report_type = st.selectbox(
            "Select Report Type",
            ["Executive Summary", "Detailed Analysis", "Performance Report", "Geographic Report"]
        )
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=df['timestamp'].min().date())
        with col2:
            end_date = st.date_input("End Date", value=df['timestamp'].max().date())
        
        # Filter data by date range
        filtered_df = df[
            (df['timestamp'].dt.date >= start_date) & 
            (df['timestamp'].dt.date <= end_date)
        ]
        
        # Generate report based on type
        if st.button("Generate Report"):
            if report_type == "Executive Summary":
                self._generate_executive_summary(filtered_df)
            elif report_type == "Detailed Analysis":
                self._generate_detailed_analysis(filtered_df)
            elif report_type == "Performance Report":
                self._generate_performance_report(filtered_df)
            elif report_type == "Geographic Report":
                self._generate_geographic_report(filtered_df)
        
        # Export options
        st.subheader("📤 Export Data")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("Export CSV"):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"incidents_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            if st.button("Export Summary"):
                summary_data = self._create_summary_data(filtered_df)
                summary_csv = summary_data.to_csv(index=False)
                st.download_button(
                    label="Download Summary",
                    data=summary_csv,
                    file_name=f"summary_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
        
        with export_col3:
            if st.button("Export Charts"):
                st.info("Chart export functionality would be implemented here")
    
    def _generate_executive_summary(self, df: pd.DataFrame):
        """Generate executive summary report"""
        st.markdown("## 📊 Executive Summary Report")
        
        total_incidents = len(df)
        critical_incidents = len(df[df['urgency'] == 'CRITICAL'])
        avg_response_time = df['response_time'].mean()
        total_affected = df['affected_population'].sum()
        
        summary_text = f"""
        ### Key Metrics
        - **Total Incidents:** {total_incidents:,}
        - **Critical Incidents:** {critical_incidents} ({critical_incidents/total_incidents*100:.1f}%)
        - **Average Response Time:** {avg_response_time:.1f} minutes
        - **Total People Affected:** {total_affected:,}
        
        ### Top Incident Types
        """
        
        top_types = df['disaster_type'].value_counts().head(3)
        for disaster_type, count in top_types.items():
            percentage = count / total_incidents * 100
            summary_text += f"- **{disaster_type.title()}:** {count} incidents ({percentage:.1f}%)\n"
        
        st.markdown(summary_text)
    
    def _generate_detailed_analysis(self, df: pd.DataFrame):
        """Generate detailed analysis report"""
        st.markdown("## 📈 Detailed Analysis Report")
        
        # Show detailed statistics
        st.dataframe(df.describe())
        
        # Show correlation matrix
        numeric_cols = ['severity', 'response_time', 'affected_population', 'hour']
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    def _generate_performance_report(self, df: pd.DataFrame):
        """Generate performance report"""
        st.markdown("## ⚡ Performance Report")
        
        # Performance by disaster type
        performance_summary = df.groupby('disaster_type').agg({
            'response_time': ['mean', 'std', 'min', 'max'],
            'severity': 'mean',
            'affected_population': 'sum'
        }).round(2)
        
        st.dataframe(performance_summary)
    
    def _generate_geographic_report(self, df: pd.DataFrame):
        """Generate geographic report"""
        st.markdown("## 🗺️ Geographic Report")
        
        # Geographic statistics
        geo_stats = {
            'Total Incidents': len(df),
            'Geographic Spread (Lat)': f"{df['lat'].max() - df['lat'].min():.4f}°",
            'Geographic Spread (Lng)': f"{df['lng'].max() - df['lng'].min():.4f}°",
            'Center Point': f"({df['lat'].mean():.4f}, {df['lng'].mean():.4f})",
            'Northernmost': f"{df['lat'].max():.4f}",
            'Southernmost': f"{df['lat'].min():.4f}",
            'Easternmost': f"{df['lng'].max():.4f}",
            'Westernmost': f"{df['lng'].min():.4f}"
        }
        
        for key, value in geo_stats.items():
            st.write(f"**{key}:** {value}")
    
    def _create_summary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary data for export"""
        summary = df.groupby('disaster_type').agg({
            'incident_id': 'count',
            'severity': ['mean', 'max'],
            'response_time': 'mean',
            'affected_population': 'sum',
            'urgency': lambda x: (x == 'CRITICAL').sum()
        }).round(2)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        return summary.reset_index()

# Usage example and testing
def test_analytics_panel():
    """Test the analytics panel component"""
    import random
    from datetime import datetime, timedelta
    
    st.title("Analytics Panel Test")
    
    # Generate sample data
    sample_incidents = []
    disaster_types = ['fire', 'flood', 'earthquake', 'accident', 'medical']
    urgency_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    
    for i in range(100):
        timestamp = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        incident = {
            'incident_id': f"INC_{i+1:03d}",
            'timestamp': timestamp.isoformat(),
            'disaster_type': random.choice(disaster_types),
            'severity': random.uniform(1, 10),
            'urgency': random.choice(urgency_levels),
            'status': random.choice(['PROCESSING', 'DISPATCHED', 'ON_SCENE', 'RESOLVED']),
            'estimated_affected_population': random.randint(0, 500),
            'estimated_response_time': random.randint(5, 30),
            'location': {
                'lat': 37.7749 + random.uniform(-0.1, 0.1),
                'lng': -122.4194 + random.uniform(-0.1, 0.1)
            },
            'text_content': f"Sample incident description {i+1}",
            'affected_infrastructure': random.sample(['roads', 'buildings', 'utilities'], random.randint(0, 3))
        }
        sample_incidents.append(incident)
    
    # Create and render analytics panel
    analytics_panel = AnalyticsPanel()
    analytics_panel.render_analytics_dashboard(sample_incidents)

if __name__ == "__main__":
    test_analytics_panel()