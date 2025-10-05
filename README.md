# Disaster Response Multi-Modal AI Agent

## Overview

Disaster Response Multi-Modal AI Agent is a state-of-the-art emergency management platform designed to accelerate and automate disaster response using advanced artificial intelligence. Integrating computer vision, natural language processing, and multi-modal data fusion, our system delivers real-time incident reporting, damage assessment, and emergency action planning — all with sub-minute response times.

## Features

- **Real-time Incident Reporting Dashboard** powered by Streamlit for intuitive user experience
- **AI-Driven Damage Assessment** using Cerebras Vision AI for ultra-fast and highly accurate image analysis
- **Intelligent Action Planning** via Google Gemini LLMs, generating prioritized emergency response steps
- **FastAPI-based MCP Gateway** orchestrating multiple AI services, ensuring scalable, reliable, and fault-tolerant operation
- **Geospatial Interactive Visualization** with incident mapping and severity indicators
- **Background Task Processing** for seamless asynchronous emergency coordination
- **Multi-Source Data Integration** including text reports, uploaded images, location data, and social media feeds

## Tech Stack & Architecture

- **FastAPI**: Serves as the MCP Gateway coordinating AI inference services and handling real-time API requests.
- **Streamlit**: UI for field agents and emergency responders to report incidents and monitor active events live.
- **Cerebras Cloud API**: Provides rapid vision intelligence for damage recognition and severity scoring.
- **Google Gemini**: Powers the emergency action generator for dynamic response plan creation.
- **Python Asyncio**: Enables scalable concurrency and efficient background task management.
- **Docker MCP Toolkit**: Implemented microservices architecture using Docker MCP Gateway principles for fault tolerance and horizontal scalability.

## Installation & Setup

1. Clone this repository:
