Disaster Response Multi-Modal AI Agent
What is this project?
Imagine there is a natural disaster, like a flood, fire, or earthquake, and people need help fast. This project is a smart system that uses artificial intelligence (AI) to help emergency teams know where disasters are happening, how bad they are, and what actions to take — all in real-time. It’s like having a super-smart assistant who can see pictures, read reports, and come up with the best emergency plans instantly.

How does it work?
User-Friendly Dashboard: The system has an easy-to-use webpage where anyone (like rescue teams or government officials) can report emergencies by typing details, sharing the location, or uploading pictures.

AI that Sees and Understands: When a disaster image is uploaded, the system uses a special AI model from Cerebras, which is very fast and smart, to analyze the damage in the pictures.

AI that Plans: Then, it uses another AI model, Google Gemini, to suggest the best emergency actions to respond to that disaster.

Fast and Reliable: Behind the scenes, a powerful engine called the MCP Gateway connects all these AI tasks so they work smoothly and quickly.

Live Map & Status: You can watch active disaster reports on a real-time map that shows where help is needed the most.

Technology we used
Streamlit: For building the interactive website. It’s simple and perfect for dashboards.

FastAPI: This is like the brain that handles all communication between the website and AI services.

Cerebras Cloud API: This AI helps analyze disaster images very fast with great accuracy.

Google Gemini: This AI generates smart emergency plans based on the analyzed data.

Docker MCP Toolkit: We used this to make our system modular and reliable, so when more incidents come in, our system can handle them all without slowing down.

Why is this project important?
When disasters strike, every second counts. Our system reduces how long it takes for emergency teams to get information and decide what to do. It helps save lives by quickly turning data into clear plans. Traditional systems are slower and rely heavily on people manually deciding what’s important. Our AI-powered system automates that, so help reaches people faster.

How to try it yourself
Clone the project from GitHub:

text
git clone <your-repository-link>
cd disaster-response-agent
Create and activate Python environment:

text
conda create -n disaster-env python=3.9 -y
conda activate disaster-env
Install required libraries:

text
pip install -r requirements.txt
Make sure you set your Cerebras API key in environment variables.

Run the backend AI server (MCP Gateway):

text
python -m uvicorn src.orchestrator.mcp_server:app --host 0.0.0.0 --port 8080 --reload
Start the dashboard:

text
streamlit run src/ui/dashboard.py --server.address localhost --server.port 8501
Open your browser to:

text
http://localhost:8501
How we used sponsor technologies
Cerebras
We integrated Cerebras Cloud API for disaster image analysis. This AI analyzes pictures of disaster damage almost instantly. It helps emergency teams understand how severe the damage is and what urgent help might be needed. Without this fast and accurate vision intelligence, emergency response would be much slower.

Docker
Our system is built on Docker’s MCP Gateway toolkit. This allows us to connect many AI services (like image analysis and action planning) smoothly and reliably. We designed it so emergency requests can be processed in parallel, making the system ready for real-world heavy use, like during floods or fires involving thousands of incidents.

Meta
We did not use Meta technologies in this project.

What we learned
Building this project taught us how to combine multiple AI models into one reliable system, manage real-time data flows, and create interfaces that non-technical users can easily use in emergencies. We learned how critical it is to build reliable, fast, and accurate decision-support tools for real-life disaster situations.

This project isn’t just about technology — it’s about making a real-world difference when people need help the most. We’re proud to contribute a system that can save lives and make disaster response smarter and faster.

If you want to learn more or try it yourself, check out our GitHub, and dive into the future of AI-powered emergency response.
