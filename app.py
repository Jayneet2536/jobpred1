
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from faker import Faker
from sklearn.ensemble import RandomForestClassifier
import random

# ------------------------
# Custom CSS Loader
# ------------------------
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error("CSS file not found.")

# ------------------------
# Data Generation and Caching
# ------------------------
@st.cache_data
def generate_data():
    fake = Faker()
    regions = ['Urban', 'Rural', 'Underrepresented']
    
    # Admission Data
    admissions = pd.DataFrame({
        'student_id': [fake.uuid4() for _ in range(1000)],
        'score': np.random.normal(70, 15, 1000).clip(0, 100),
        'region': np.random.choice(regions, 1000),
        'institution_capacity': np.random.randint(50, 500, 1000),
        'admitted': np.random.choice([0, 1], 1000, p=[0.3, 0.7])
    })
    
    # Expanded Job Market Roles and Requirements
    roles = {
        'Data Engineer': {'Python': 4, 'SQL': 4, 'Cloud': 4, 'ETL': 3},
        'AI Specialist': {'Python': 5, 'ML': 5, 'TensorFlow': 4},
        'Cloud Architect': {'Cloud': 5, 'Networking': 4, 'Security': 4},
        'Data Scientist': {'Python': 5, 'ML': 5, 'SQL': 4, 'Statistics': 4},
        'DevOps Engineer': {'Cloud': 5, 'Python': 4, 'Linux': 4, 'CI/CD': 4},
        'Cybersecurity Analyst': {'Networking': 5, 'Security': 5, 'Python': 3, 'Risk Assessment': 4},
        'Full-Stack Developer': {'Python': 4, 'JavaScript': 5, 'React': 4, 'SQL': 3},
        'Product Manager': {'Communication': 5, 'Agile': 4, 'Analytics': 4, 'Leadership': 5},
    }
    
    # Job Market Data for all roles
    job_data = pd.DataFrame({
        'role': list(roles.keys()),
        'avg_salary': [95000, 120000, 110000, 115000, 105000, 100000, 98000, 130000],
        'growth_rate': [15, 25, 20, 22, 18, 20, 17, 23],
        'demand': [85, 90, 80, 88, 82, 87, 83, 89]
    })
    
    # Expanded Courses Data
    courses = pd.DataFrame({
        'skill': ['Python', 'Cloud', 'ML', 'SQL', 'TensorFlow', 'ETL', 'Statistics', 
                  'Linux', 'CI/CD', 'Networking', 'Security', 'Risk Assessment', 
                  'JavaScript', 'React', 'Communication', 'Agile', 'Analytics', 'Leadership'],
        'name': [
            'Advanced Python', 'AWS Certified', 'ML Bootcamp', 'SQL Mastery', 'TensorFlow Pro', 
            'ETL Fundamentals', 'Statistics for Data Science', 'Linux Administration', 'CI/CD with Jenkins',
            'Networking Essentials', 'Cybersecurity Fundamentals', 'Risk Management in IT',
            'JavaScript Deep Dive', 'React from Scratch', 'Effective Communication', 'Agile Project Management',
            'Data Analytics with Python', 'Leadership Excellence'
        ],
        'duration': ['6w', '8w', '10w', '4w', '6w', '5w', '7w', '6w', '4w', '5w', '7w', '5w', '6w', '6w', '4w', '5w', '6w', '6w'],
        'platform': ['Coursera', 'Udacity', 'edX', 'Udemy', 'Pluralsight',
                     'LinkedIn Learning', 'Coursera', 'Udacity', 'Pluralsight', 'edX', 'Udemy', 'LinkedIn Learning',
                     'Udacity', 'Coursera', 'LinkedIn Learning', 'edX', 'Udemy', 'Coursera']
    })
    
    # Simulated Career News Data
    news = pd.DataFrame({
        'headline': [
            "Tech Giants Shift Focus to AI-Driven Solutions",
            "Remote Work Revolutionizes Tech Industry",
            "New Certification Programs Gain Popularity Among Engineers",
            "Cybersecurity: Top Threats and How to Combat Them",
            "The Rise of Full-Stack Developers in the Modern Era",
            "Innovations in Cloud Computing: What to Expect Next",
            "Agile Methodologies: Transforming Project Management"
        ],
        'link': [
            "https://news.example.com/ai-solutions",
            "https://news.example.com/remote-work",
            "https://news.example.com/new-certifications",
            "https://news.example.com/cyber-threats",
            "https://news.example.com/full-stack-rise",
            "https://news.example.com/cloud-innovations",
            "https://news.example.com/agile-transformation"
        ]
    })
    
    return admissions, roles, job_data, courses, news

# ------------------------
# Machine Learning Model Training
# ------------------------
def train_models(admissions):
    X = admissions[['score', 'institution_capacity']]
    y = admissions['admitted']
    admission_model = RandomForestClassifier().fit(X, y)
    return admission_model

# ------------------------
# Visualizations
# ------------------------
def create_radar_chart(scores):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=['Salary', 'Skills', 'Chances', 'Stability', 'Trust', 'Growth'],
        fill='toself',
        line=dict(color='#4287f5'),
        name='Career Potential'
    ))
    fig.update_layout(
        showlegend=False,
        height=400,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        )
    )
    return fig

def create_career_timeline(role_data, timeline):
    years = list(range(1, timeline + 1))
    progression = [min(100, role_data['demand'] + role_data['growth_rate'] * year * 0.5) for year in years]
    fig = px.line(
        x=years,
        y=progression,
        markers=True,
        labels={'x': 'Years', 'y': 'Career Potential Score'},
        title="Projected Career Growth Over Time"
    )
    fig.update_layout(height=300)
    return fig

def job_market_insights(job_data):
    fig = px.bar(job_data, x='role', y='demand', color='growth_rate',
                 labels={'demand': 'Market Demand (%)', 'growth_rate': 'Growth Rate (%)'},
                 title="Job Market Demand & Growth by Role")
    fig.update_layout(height=400)
    st.plotly_chart(fig)

# ------------------------
# Additional Interactive Features
# ------------------------
def mentor_recommendations(experience):
    if experience < 2:
        mentor = "Junior Mentor: Look for industry peers or recent grads to guide you through early challenges."
    elif experience < 5:
        mentor = "Mid-Level Mentor: Connect with professionals with 5-10 years of experience for mentorship."
    else:
        mentor = "Senior Mentor: Seek out industry leaders or executives to refine your career strategy."
    return mentor

def interview_simulator(role):
    # Simulated interview questions for each role
    questions_bank = {
        'Data Engineer': [
            "Explain the ETL process and its challenges.",
            "How do you optimize SQL queries?",
            "Describe a time you handled data pipeline failures."
        ],
        'AI Specialist': [
            "What are the differences between supervised and unsupervised learning?",
            "Explain the concept of overfitting in neural networks.",
            "How do you choose the right model architecture for a problem?"
        ],
        'Cloud Architect': [
            "How do you design a scalable cloud solution?",
            "What are the key differences between IaaS, PaaS, and SaaS?",
            "Discuss your experience with multi-cloud environments."
        ],
        'Data Scientist': [
            "How do you handle missing data in a dataset?",
            "What is the role of feature engineering in machine learning?",
            "Explain a project where you implemented statistical models."
        ],
        'DevOps Engineer': [
            "What are the best practices for CI/CD?",
            "How do you manage container orchestration?",
            "Describe your experience with infrastructure as code."
        ],
        'Cybersecurity Analyst': [
            "What are the most common cybersecurity threats today?",
            "How do you approach risk assessment?",
            "Describe your experience with incident response."
        ],
        'Full-Stack Developer': [
            "Explain the MVC architecture in web development.",
            "How do you ensure the security of a web application?",
            "Describe a challenging bug you fixed on the front-end."
        ],
        'Product Manager': [
            "How do you prioritize features for a product roadmap?",
            "Explain a time when you had to make a tough product decision.",
            "How do you handle stakeholder communication?"
        ]
    }
    questions = questions_bank.get(role, ["No interview questions available for this role."])
    st.markdown("#### Interview Simulation")
    for q in questions:
        st.markdown(f"- {q}")

def create_skill_network(roles, courses):
    # Build a network graph from skills required for each role
    G = nx.Graph()
    # Add nodes for each role and their required skills
    for role, skills in roles.items():
        G.add_node(role, type='role')
        for skill in skills.keys():
            G.add_node(skill, type='skill')
            G.add_edge(role, skill)
    # Create a plotly network graph
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_type = G.nodes[node]['type']
        node_color.append('#FF5733' if node_type=='role' else '#33C1FF')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=15,
            line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Skill & Role Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500))
    st.plotly_chart(fig)

# ------------------------
# Main Application
# ------------------------
def main():
    load_css()
    st.title("🎓 Engineering Career Navigator - Ultimate Edition")
    
    # Generate Data and Train Model
    admissions, roles, job_data, courses, news = generate_data()
    admission_model = train_models(admissions)
    
    # Sidebar Navigation with multiple pages
    page = st.sidebar.radio("Navigation", [
        "🏠 Home", 
        "🎯 Career Predictor", 
        "📚 Learning Path", 
        "📊 Job Market Insights", 
        "💬 Interview Simulator", 
        "🔗 Skill Network", 
        "📰 Career News"
    ])
    
    # ------------------------
    # Home Page
    # ------------------------
    if page == "🏠 Home":
        st.header("Welcome to the Ultimate Engineering Career Navigator")
        st.image("https://cdn.pixabay.com/photo/2018/09/27/09/22/artificial-intelligence-3706562_1280.jpg", use_container_width=True)
        st.markdown("""
        *Discover your perfect career path with:*
        - Dynamic role and skills prediction
        - Personalized learning pathways
        - Interactive market insights
        - Simulated interview preparation
        - Real-time skill network visualizations
        - Curated career news and trends
        """)
    
    # ------------------------
    # Career Predictor Page
    # ------------------------
    elif page == "🎯 Career Predictor":
        st.header("🧭 Career Prediction Engine")
        
        with st.form("career_form"):
            col1, col2 = st.columns(2)
            with col1:
                degree = st.selectbox("Highest Degree", ["Bachelor's", "Master's", "PhD"])
                experience = st.slider("Years of Experience", 0, 30, 2)
                region = st.selectbox("Preferred Region", ['Urban', 'Rural', 'Underrepresented'])
            with col2:
                target_role = st.selectbox("Target Role", list(roles.keys()))
                st.markdown("*Skill Self-Assessment*")
                # Basic sliders for common skills
                python = st.slider("Python", 1, 5, 3)
                ml = st.slider("Machine Learning", 1, 5, 2)
                cloud = st.slider("Cloud Computing", 1, 5, 2)
                # Additional sliders dynamically generated based on role requirements
                extra_skills = {}
                additional_skills = set(roles[target_role].keys()) - {'Python', 'Machine Learning', 'Cloud'}
                for skill in additional_skills:
                    default = 3
                    extra_skills[skill] = st.slider(f"{skill}", 1, 5, default)
            submitted = st.form_submit_button("🚀 Analyze Career Potential")
            
        if submitted:
            user_skills = {'Python': python, 'Machine Learning': ml, 'Cloud Computing': cloud}
            user_skills.update(extra_skills)
            required = roles[target_role]
            gaps = {skill: req - user_skills.get(skill, 0)
                    for skill, req in required.items() if user_skills.get(skill, 0) < req}
            
            role_data = job_data[job_data['role'] == target_role].iloc[0]
            metrics = {
                'Salary': role_data['avg_salary'] / 2000,
                'Skills': max(0, 100 - (sum(gaps.values()) / (len(required) * 5)) * 100),
                'Chances': min(100, role_data['demand'] * 0.8),
                'Stability': 75,
                'Trust': 85,
                'Growth': role_data['growth_rate']
            }
            
            st.subheader(f"📊 Career Potential for {target_role}")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.plotly_chart(create_radar_chart(list(metrics.values())))
            with col2:
                st.metric("Average Salary", f"${role_data['avg_salary']:,.0f}")
                st.metric("Market Demand", f"{role_data['demand']}%")
                st.metric("Growth Rate", f"{role_data['growth_rate']}% YoY")
                st.info(mentor_recommendations(experience))
            
            st.subheader("📈 Career Timeline Projection")
            timeline_years = st.slider("Projection Years", 1, 30, 5)
            st.plotly_chart(create_career_timeline(role_data, timeline_years))
            
            st.subheader("🔍 Skill Development Recommendations")
            if gaps:
                for skill, gap in sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:3]:
                    relevant_courses = courses[courses['skill'] == skill]
                    with st.expander(f"🚨 {skill} (Improve by {gap} level{'s' if gap>1 else ''})"):
                        for _, course in relevant_courses.iterrows():
                            st.markdown(f"""
                            *{course['name']}*  
                            - Platform: {course['platform']}  
                            - Duration: {course['duration']}  
                            """)
            else:
                st.success("Your skills exceed the requirements for this role!")
    
    # ------------------------
    # Learning Path Page
    # ------------------------
    elif page == "📚 Learning Path":
        st.header("🛠 Skill Development Roadmap")
        role = st.selectbox("Select Target Role", list(roles.keys()))
        if role:
            st.markdown("### Generate a Full Roadmap for Your Target Goal")
            timeline = st.slider("Select your desired timeline (in months)", 6, 36, 12)
            focus_area = st.multiselect(
                "Select focus areas", 
                options=["Technical Skills", "Soft Skills", "Certifications"], 
                default=["Technical Skills"]
            )
            
            # Define a detailed roadmap for each target role
            roadmap_details = {
                "Data Engineer": [
                    {
                        "Milestone": "Foundations",
                        "Focus Skill": "Python & SQL",
                        "Action": "Learn the fundamentals of programming with Python and basic database management using SQL.",
                        "Recommended Course": "SQL Mastery on Udemy (4w)"
                    },
                    {
                        "Milestone": "Intermediate ETL & Cloud",
                        "Focus Skill": "ETL & Cloud",
                        "Action": "Master data pipeline techniques and understand cloud storage solutions.",
                        "Recommended Course": "ETL Fundamentals on LinkedIn Learning (5w)"
                    },
                    {
                        "Milestone": "Advanced Data Engineering",
                        "Focus Skill": "Big Data & Cloud Scaling",
                        "Action": "Apply advanced techniques for processing big data on cloud platforms.",
                        "Recommended Course": "AWS Certified on Udacity (8w)"
                    }
                ],
                "AI Specialist": [
                    {
                        "Milestone": "Foundations",
                        "Focus Skill": "Python & Intro to ML",
                        "Action": "Build a solid foundation in Python and learn the basics of machine learning.",
                        "Recommended Course": "ML Bootcamp on edX (10w)"
                    },
                    {
                        "Milestone": "Deep Learning",
                        "Focus Skill": "TensorFlow & Advanced ML",
                        "Action": "Dive deeper into neural networks, deep learning concepts, and TensorFlow.",
                        "Recommended Course": "TensorFlow Pro on Pluralsight (6w)"
                    },
                    {
                        "Milestone": "AI Deployment",
                        "Focus Skill": "Model Deployment & Optimization",
                        "Action": "Learn to deploy and optimize AI models in production environments.",
                        "Recommended Course": "AI Deployment Strategies on Coursera (8w)"
                    }
                ],
                "Cloud Architect": [
                    {
                        "Milestone": "Cloud Fundamentals",
                        "Focus Skill": "Cloud Concepts",
                        "Action": "Understand cloud computing basics, including IaaS, PaaS, and SaaS.",
                        "Recommended Course": "AWS Certified on Udacity (8w)"
                    },
                    {
                        "Milestone": "Networking & Security",
                        "Focus Skill": "Advanced Networking & Security",
                        "Action": "Learn advanced networking architectures and cloud security best practices.",
                        "Recommended Course": "Cybersecurity Fundamentals on Udemy (7w)"
                    },
                    {
                        "Milestone": "Architecture Mastery",
                        "Focus Skill": "Designing Scalable Systems",
                        "Action": "Design and implement scalable, robust cloud architectures.",
                        "Recommended Course": "Cloud Architect Pro on edX (10w)"
                    }
                ],
                "Data Scientist": [
                    {
                        "Milestone": "Data Analysis",
                        "Focus Skill": "Python & Statistics",
                        "Action": "Learn data analysis, visualization, and statistical fundamentals.",
                        "Recommended Course": "Advanced Python on Coursera (6w)"
                    },
                    {
                        "Milestone": "Machine Learning",
                        "Focus Skill": "ML Algorithms",
                        "Action": "Develop proficiency in machine learning techniques and model building.",
                        "Recommended Course": "ML Bootcamp on edX (10w)"
                    },
                    {
                        "Milestone": "Real-World Projects",
                        "Focus Skill": "Applied Data Science",
                        "Action": "Work on real-world projects to solidify your data science skills.",
                        "Recommended Course": "Data Science Capstone on Udacity (8w)"
                    }
                ],
                "DevOps Engineer": [
                    {
                        "Milestone": "Foundations",
                        "Focus Skill": "Linux & Scripting",
                        "Action": "Master Linux system administration and scripting with Python.",
                        "Recommended Course": "Linux Administration on Udacity (6w)"
                    },
                    {
                        "Milestone": "CI/CD & Automation",
                        "Focus Skill": "Automation Tools",
                        "Action": "Learn best practices for continuous integration, deployment, and automation.",
                        "Recommended Course": "CI/CD with Jenkins on Pluralsight (4w)"
                    },
                    {
                        "Milestone": "Cloud & Containerization",
                        "Focus Skill": "Cloud Orchestration",
                        "Action": "Implement container orchestration and cloud deployment strategies.",
                        "Recommended Course": "AWS Certified on Udacity (8w)"
                    }
                ],
                "Cybersecurity Analyst": [
                    {
                        "Milestone": "Cybersecurity Basics",
                        "Focus Skill": "Networking & Security",
                        "Action": "Learn the fundamentals of cybersecurity, including threat types and prevention.",
                        "Recommended Course": "Cybersecurity Fundamentals on Udemy (7w)"
                    },
                    {
                        "Milestone": "Advanced Threat Analysis",
                        "Focus Skill": "Risk Assessment",
                        "Action": "Deep dive into threat detection and risk management techniques.",
                        "Recommended Course": "Risk Management in IT on LinkedIn Learning (5w)"
                    },
                    {
                        "Milestone": "Incident Response",
                        "Focus Skill": "Crisis Management",
                        "Action": "Prepare for real-world incident response and recovery scenarios.",
                        "Recommended Course": "Incident Response Strategies on Coursera (8w)"
                    }
                ],
                "Full-Stack Developer": [
                    {
                        "Milestone": "Front-End Fundamentals",
                        "Focus Skill": "JavaScript & React",
                        "Action": "Master front-end development with JavaScript and modern frameworks like React.",
                        "Recommended Course": "JavaScript Deep Dive on Udacity (6w)"
                    },
                    {
                        "Milestone": "Back-End Development",
                        "Focus Skill": "Python & SQL",
                        "Action": "Learn server-side programming and database management.",
                        "Recommended Course": "Advanced Python on Coursera (6w)"
                    },
                    {
                        "Milestone": "Full-Stack Integration",
                        "Focus Skill": "Application Deployment",
                        "Action": "Build and deploy full-stack applications from scratch.",
                        "Recommended Course": "React from Scratch on Coursera (6w)"
                    }
                ],
                "Product Manager": [
                    {
                        "Milestone": "Foundations",
                        "Focus Skill": "Communication & Analytics",
                        "Action": "Develop core skills in effective communication and data-driven decision making.",
                        "Recommended Course": "Effective Communication on LinkedIn Learning (4w)"
                    },
                    {
                        "Milestone": "Agile & Leadership",
                        "Focus Skill": "Project Management",
                        "Action": "Learn agile methodologies and how to lead cross-functional teams.",
                        "Recommended Course": "Agile Project Management on edX (5w)"
                    },
                    {
                        "Milestone": "Strategic Planning",
                        "Focus Skill": "Product Strategy",
                        "Action": "Master strategic planning and stakeholder management for successful product launches.",
                        "Recommended Course": "Leadership Excellence on Coursera (6w)"
                    }
                ]
            }
            
            # Retrieve the roadmap for the selected role.
            roadmap = roadmap_details.get(role, [])
            st.subheader(f"📅 {timeline}-Month Roadmap for {role}")
            
            # Adjust the roadmap if the user's timeline allows for more milestones.
            total_milestones = len(roadmap)
            if timeline // 3 > total_milestones:
                st.info(f"Your selected timeline allows for more milestones than available. Showing the full roadmap for {role}.")
            
            # Display each milestone in the roadmap.
            for milestone in roadmap:
                st.markdown(f"### {milestone['Milestone']}: {milestone['Focus Skill']}")
                st.write(f"*Action:* {milestone['Action']}")
                st.markdown(f"*Recommended Course:* {milestone['Recommended Course']}")
                st.markdown("---")
            
            # Optionally add a final certification milestone if Certifications is in focus.
            if "Certifications" in focus_area:
                st.markdown("### Final Milestone: Certification")
                st.write("*Action:* Prepare for and obtain an industry-recognized certification to validate your skills and boost your profile.")
                st.markdown("*Recommended:* Explore certification programs on Coursera, Udacity, or edX.")
                st.markdown("---")
            
            # Summary Table of the Roadmap
            st.subheader("Roadmap Summary")
            summary = []
            for milestone in roadmap:
                summary.append({
                    "Milestone": milestone["Milestone"],
                    "Focus Skill": milestone["Focus Skill"],
                    "Action": milestone["Action"],
                    "Recommended Course": milestone["Recommended Course"]
                })
            if "Certifications" in focus_area:
                summary.append({
                    "Milestone": "Final Milestone",
                    "Focus Skill": "Certification",
                    "Action": "Prepare for and obtain an industry-recognized certification.",
                    "Recommended Course": "Certification programs on Coursera/Udacity/edX"
                })
            roadmap_df = pd.DataFrame(summary)
            st.dataframe(roadmap_df)
            
            # ------------------------
            # Insert the Floating Gemini Chatbot Widget
            # ------------------------
            chatbot_html = """
            <div id="chatbot-container">
            <div id="chatbot-icon" onclick="toggleChat()">💬</div>
            <div id="chatbot-window">
                <div id="chatbot-header">Gemini Chatbot <span onclick="toggleChat()" style="cursor:pointer;">✖</span></div>
                <div id="chatbot-messages"></div>
                <input type="text" id="chatbot-input" placeholder="Ask for recommendations..." onkeypress="handleKeyPress(event)">
            </div>
            </div>
            <style>
            #chatbot-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 1000;
            }
            #chatbot-icon {
                width: 60px;
                height: 60px;
                background-color: #4287f5;
                border-radius: 50%;
                text-align: center;
                line-height: 60px;
                color: white;
                font-size: 30px;
                cursor: pointer;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            }
            #chatbot-window {
                display: none;
                width: 300px;
                height: 400px;
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 10px;
                position: relative;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
            }
            #chatbot-header {
                background-color: #4287f5;
                color: white;
                padding: 10px;
                border-radius: 10px 10px 0 0;
                font-weight: bold;
            }
            #chatbot-messages {
                height: 300px;
                padding: 10px;
                overflow-y: auto;
                font-size: 14px;
            }
            #chatbot-input {
                width: calc(100% - 20px);
                padding: 10px;
                border: none;
                border-top: 1px solid #ccc;
                font-size: 14px;
            }
            </style>
            
            <script>
            // Asynchronously calls the Gemini API
            async function getChatbotResponse(query) {
                // Replace with your actual Gemini API endpoint
                const apiEndpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyDMomklmZmMiNvekbkzsztLxAvX3t95NUI";
                // Replace with your actual API key
                const apiKey = "AIzaSyDMomklmZmMiNvekbkzsztLxAvX3t95NUI";
                
                try {
                    const response = await fetch(apiEndpoint, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            "Authorization": `Bearer ${apiKey}`
                        },
                        body: JSON.stringify({ prompt: query })
                    });
                    if (!response.ok) {
                        throw new Error("Network response was not ok");
                    }
                    const data = await response.json();
                    // Assume the API returns the response in a property called 'reply'
                    return data.reply || "Sorry, I didn't understand that.";
                } catch (error) {
                    console.error("Error calling Gemini API:", error);
                    return "I'm sorry, there was an error processing your request.";
                }
            }

            // Toggle the chat window display
            function toggleChat(){
                var chatWindow = document.getElementById('chatbot-window');
                if(chatWindow.style.display === 'none' || chatWindow.style.display === ''){
                    chatWindow.style.display = 'block';
                } else {
                    chatWindow.style.display = 'none';
                }
            }

            // Handles key presses in the input field and uses the Gemini API to get a response
            async function handleKeyPress(event){
                if(event.key === 'Enter'){
                    var input = document.getElementById('chatbot-input');
                    var message = input.value;
                    if(message.trim() !== ""){
                        addMessage("You: " + message);
                        input.value = "";
                        // Await the response from the Gemini API
                        const responseText = await getChatbotResponse(message);
                        addMessage("Gemini: " + responseText);
                    }
                }
            }

            // Adds a new message to the chat window
            function addMessage(msg){
                var messagesDiv = document.getElementById('chatbot-messages');
                var messageElem = document.createElement('div');
                messageElem.textContent = msg;
                messagesDiv.appendChild(messageElem);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            </script>

            
            """
            st.components.v1.html(chatbot_html, height=500)


    
    # ------------------------
    # Job Market Insights Page
    # ------------------------
    elif page == "📊 Job Market Insights":
        st.header("📊 Job Market Insights")
        st.markdown("Compare job market trends across engineering roles.")
        job_market_insights(job_data)
        st.markdown("*Detailed Data:*")
        st.dataframe(job_data)
    
    # ------------------------
    # Interview Simulator Page
    # ------------------------
    elif page == "💬 Interview Simulator":
        st.header("💬 Interview Simulator")
        selected_role = st.selectbox("Select a Role for Interview Simulation", list(roles.keys()))
        if st.button("Generate Interview Questions"):
            interview_simulator(selected_role)
    
    # ------------------------
    # Skill Network Map Page
    # ------------------------
    elif page == "🔗 Skill Network":
        st.header("🔗 Interactive Skill & Role Network")
        st.markdown("Explore the relationships between roles and required skills.")
        create_skill_network(roles, courses)
    
    # ------------------------
    # Career News Page
    # ------------------------
    elif page == "📰 Career News":
        st.header("📰 Latest Career News & Trends")
        st.markdown("Stay updated with the latest trends in the tech and engineering sectors.")
        for i, row in news.iterrows():
            st.markdown(f"[{row['headline']}]({row['link']})")
            st.write("---")

if __name__ == "__main__":
    main()