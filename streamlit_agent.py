import io
import streamlit as st
import requests
from langchain.embeddings import OpenAIEmbeddings
from anthropic import Anthropic
from langgraph.graph import StateGraph
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
import networkx as nx
import matplotlib.pyplot as plt
import pymupdf4llm
import tempfile
import os

class JobGraphState(BaseModel):
    term: str
    experience: str
    anthropic_key: str
    openai_key: str
    jobs: list = []
    embeddings: list = []
    style: list = []
    pdf_text: str = ""
    skills: list = []
    industry: str = ""


# --- GUI Layout ---
st.set_page_config(page_title="Tech Job Analyzer", layout="wide")
st.title("Tech Job Analyzer (LangGraph Demo)")

with st.sidebar:
    st.header("API Keys & Options")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    openai_key = st.text_input("OpenAI API Key (for embeddings)", type="password")
    st.markdown("---")
    st.header("Visualization")
    show_mermaid = st.checkbox("Show Mermaid Graph Visualization")

st.header("1. Upload your CV or Job Description PDF")
pdf_file = st.file_uploader("Upload a PDF", type="pdf")
pdf_text = ""
if pdf_file is not None:
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Use the temporary file path with pymupdf4llm
            pdf_text = pymupdf4llm.to_markdown(tmp_path)
            if not isinstance(pdf_text, str):
                pdf_text = str(pdf_text)
        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"PDF extraction error: {e}")  # Debug print
        st.error(f"Error extracting PDF text: {e}")
        pdf_text = ""
    if pdf_text.strip():
        st.success("PDF uploaded and text extracted.")
        st.text_area("Extracted PDF Text", pdf_text, height=200)
    else:
        st.warning("No text could be extracted from the PDF. Please try another file.")
else:
    st.info("Please upload a PDF to begin.")

st.header("2. Run Job Analyzer")
st.markdown("Click the button below to analyze your PDF and find relevant jobs.")
go = st.button("Analyze PDF & Search Jobs")

def fetch_jobs(state):
    with st.status("Step 2: Searching for relevant jobs...") as status:
        term = state.term
        url = f"https://remotive.com/api/remote-jobs?search={term}"
        try:
            resp = requests.get(url, timeout=10)
            state.jobs = resp.json().get("jobs", [])
            status.update(label=f"✅ Found {len(state.jobs)} initial job matches!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error fetching jobs: {str(e)}")
            state.jobs = []
    return state

def filter_by_experience(state):
    with st.status("Step 3: Filtering jobs by experience and skills...") as status:
        jobs = state.jobs
        level = state.experience.lower()
        skills = set(skill.lower() for skill in state.skills)
        
        filtered_jobs = []
        for job in jobs:
            job_text = (job["title"] + " " + job["description"]).lower()
            
            # Check experience level match
            level_match = (level == "all" or level in job_text)
            
            # Check for skills match
            skills_found = sum(1 for skill in skills if skill in job_text)
            skills_match = skills_found >= min(2, len(skills))  # At least 2 skills match if available
            
            # Check industry match if specified
            industry_match = True
            if state.industry:
                industry_match = state.industry.lower() in job_text
            
            if level_match and skills_match and industry_match:
                # Add a relevance score
                job["relevance_score"] = (skills_found / len(skills) if skills else 0.5)
                filtered_jobs.append(job)
        
        # Sort by relevance score
        filtered_jobs.sort(key=lambda x: x["relevance_score"], reverse=True)
        state.jobs = filtered_jobs
        
        status.update(label=f"✅ Found {len(filtered_jobs)} highly relevant positions!", state="complete", expanded=False)
    return state

def embed_jobs(state):
    jobs = state.jobs
    openai_key = state.openai_key
    if not openai_key:
        state.embeddings = [None]*len(jobs)
    else:
        embedder = OpenAIEmbeddings(openai_api_key=openai_key)
        state.embeddings = embedder.embed_documents([j["description"][:2000] for j in jobs])
    return state

def analyze_style(state):
    jobs = state.jobs
    anthropic_key = state.anthropic_key
    if not anthropic_key:
        state.style = ["No API key"]*len(jobs)
    else:
        client = Anthropic(api_key=anthropic_key)
        results = []
        for job in jobs:
            prompt = f"Analyze the following job description for writing style and redacción quality. Give a short summary and a quality score (1-10):\n\n{job['description'][:1500]}"
            try:
                response = client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                )
                results.append(response.content[0].text)
            except Exception as e:
                results.append(f"Error: {str(e)}")
        state.style = results
    return state

# --- New Node: Analyze PDF and Extract Search Term/Experience ---
def analyze_pdf(state):
    pdf_text = state.pdf_text
    anthropic_key = state.anthropic_key
    
    if not pdf_text or not anthropic_key:
        st.error("Please provide both a PDF and API key")
        state.term = "Python Developer"
        state.experience = "all"
        return state
        
    with st.status("Step 1: Analyzing PDF content...") as status:
        client = Anthropic(api_key=anthropic_key)
        prompt = """Analyze the following CV or job description text. Extract the following information:
        1. The most relevant job search terms (e.g., 'Senior Data Scientist', 'Full Stack Developer')
        2. Key technical skills
        3. Experience level (junior, mid, senior)
        4. Industry focus if any
        
        Return a JSON object with keys:
        - 'term': The primary job search term
        - 'experience': Experience level
        - 'skills': List of key technical skills
        - 'industry': Primary industry focus
        
        Here's the text to analyze:\n\n""" + pdf_text[:4000]
        
        try:
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text
            
            try:
                import json
                parsed = json.loads(result)
                state.term = parsed.get("term", "Python Developer")
                state.experience = parsed.get("experience", "all")
                state.skills = parsed.get("skills", [])
                state.industry = parsed.get("industry", "")
                
                status.update(label="✅ PDF Analysis Complete!", state="complete", expanded=False)
                st.write("📑 Extracted Profile:")
                st.write(f"- Job Focus: {state.term}")
                st.write(f"- Experience Level: {state.experience}")
                st.write(f"- Key Skills: {', '.join(state.skills[:5])}")
                if state.industry:
                    st.write(f"- Industry Focus: {state.industry}")
                    
            except Exception as e:
                st.warning("Could not parse LLM response, using fallback values")
                state.term = "Python Developer"
                state.experience = "all"
                state.skills = []
                state.industry = ""
                
        except Exception as e:
            st.error(f"Error during PDF analysis: {str(e)}")
            state.term = "Python Developer"
            state.experience = "all"
            state.skills = []
            state.industry = ""
            
    return state

# --- LangGraph definition ---
def make_graph():
    builder = StateGraph(state_schema=JobGraphState)
    builder.add_node("analyze_pdf", analyze_pdf)  # <-- NEW NODE
    builder.add_node("fetch_jobs", fetch_jobs)
    builder.add_node("filter_by_experience", filter_by_experience)
    builder.add_node("embed_jobs", embed_jobs)
    builder.add_node("analyze_style", analyze_style)
    builder.add_edge("analyze_pdf", "fetch_jobs")  # <-- NEW EDGE
    builder.add_edge("fetch_jobs", "filter_by_experience")
    builder.add_edge("filter_by_experience", "embed_jobs")
    builder.add_edge("embed_jobs", "analyze_style")
    builder.set_entry_point("analyze_pdf")  # <-- NEW ENTRY POINT
    builder.add_edge("analyze_style", END)
    graph = builder.compile()
    return graph, builder

graph, builder = make_graph()

def plot_agent_graph():
    G = nx.DiGraph()
    # Add nodes with custom shapes and colors for vertical flow
    G.add_node("fetch_jobs", label="Fetch Jobs", shape="rounded", fillcolor="#E3F2FD")
    G.add_node("filter_by_experience", label="Filter by Experience", shape="rounded", fillcolor="#E8F5E9")
    G.add_node("embed_jobs", label="Embed Jobs", shape="rounded", fillcolor="#FFFDE7")
    G.add_node("analyze_style", label="Analyze Style", shape="rounded", fillcolor="#FFEBEE")
    G.add_node("END", label="END", shape="rectangle", fillcolor="#ECEFF1")
    # Edges for vertical flow
    G.add_edges_from([
        ("fetch_jobs", "filter_by_experience"),
        ("filter_by_experience", "embed_jobs"),
        ("embed_jobs", "analyze_style"),
        ("analyze_style", "END"),
    ])
    # Arrange nodes vertically
    pos = {
        "fetch_jobs": (0, 4),
        "filter_by_experience": (0, 3),
        "embed_jobs": (0, 2),
        "analyze_style": (0, 1),
        "END": (0, 0),
    }
    fig, ax = plt.subplots(figsize=(4, 7))
    node_colors = [G.nodes[n].get("fillcolor", "#FFFFFF") for n in G.nodes]
    node_labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5000, edgecolors="#333333", linewidths=2, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=30, edge_color="#888888", width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=14, font_weight="bold", font_color="#222222", ax=ax)
    ax.set_axis_off()
    plt.tight_layout()
    # st.pyplot(fig)
    # st.pyplot(fig)
    # Save figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, width=500)

st.header("Agent Workflow Graph")
plot_agent_graph()

if go and anthropic_key:
    with st.spinner("Running pipeline..."):
        initial_state = {
            "term": "",  # will be overwritten by analyze_pdf
            "experience": "",  # will be overwritten by analyze_pdf
            "anthropic_key": anthropic_key,
            "openai_key": openai_key,
            "jobs": [],
            "embeddings": [],
            "style": [],
            "pdf_text": pdf_text or "",  # Always pass a string
            "skills": [],
            "industry": ""
        }
        result_state = graph.invoke(initial_state)
        jobs = result_state["jobs"]
        embeddings = result_state["embeddings"]
        style = result_state["style"]
        
        # Add embeddings and style analysis to jobs
        for idx, job in enumerate(jobs):
            job["embedding"] = embeddings[idx] if embeddings else None
            job["style_analysis"] = style[idx] if idx < len(style) else ""
        
        if len(jobs) == 0:
            st.warning("No matching jobs found. Try adjusting your PDF content or adding more details to your CV.")
        else:
            st.success(f"🎯 Found {len(jobs)} matching positions!")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Quick View", "📑 Detailed View"])
            
            with tab1:
                # Compact view with key information
                for i, job in enumerate(jobs[:10], 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### {i}. {job['title']}")
                            st.markdown(f"**Company:** {job['company_name']}")
                            st.markdown(f"**Location:** {job['candidate_required_location']}")
                            match_score = int(job.get('relevance_score', 0.5) * 100)
                            st.progress(job.get('relevance_score', 0.5), text=f"Match Score: {match_score}%")
                        with col2:
                            st.link_button("Apply Now", job['url'], use_container_width=True)
                    st.divider()
            
            with tab2:
                # Detailed view with full descriptions
                for i, job in enumerate(jobs[:10], 1):
                    with st.expander(f"{i}. {job['title']} at {job['company_name']}"):
                        st.markdown("### Job Details")
                        st.markdown(f"**Location:** {job['candidate_required_location']}")
                        st.markdown(f"**Job Type:** {job.get('job_type', 'Not specified')}")
                        
                        # Style analysis if available
                        if job.get('style_analysis'):
                            st.markdown("### Writing Style Analysis")
                            st.info(job['style_analysis'])
                        
                        st.markdown("### Full Description")
                        st.markdown(job["description"])
                        
                        st.link_button("Apply for this Position", job['url'], use_container_width=True)
else:
    st.info("👋 Welcome! To get started:\n\n" + 
            "1. Enter your API keys in the sidebar\n" +
            "2. Upload your CV or a job description PDF\n" +
            "3. Click 'Analyze PDF & Search Jobs' to find matching positions")