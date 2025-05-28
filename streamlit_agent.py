import io
import streamlit as st
import requests
from langchain_openai import OpenAIEmbeddings
from anthropic import Anthropic
from langgraph.graph import StateGraph
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
import networkx as nx
import matplotlib.pyplot as plt
import pymupdf4llm
import tempfile
import os

# Must be the first Streamlit command
st.set_page_config(page_title="Tech Job Analyzer", layout="wide")

@st.cache_data(ttl=3600)
def load_job_categories():
    """Load and cache job categories from Remotive API"""
    try:
        resp = requests.get("https://remotive.com/api/remote-jobs/categories", timeout=10)
        resp.raise_for_status()
        categories = resp.json().get("jobs", [])
        # Convert to simple list of category names
        category_names = [cat["name"] for cat in categories]
        return category_names
    except Exception as e:
        st.error(f"Error loading job categories: {e}")
        return []

# Add categories to the state model
class JobGraphState(BaseModel):
    anthropic_key: str
    openai_key: str
    pdf_text: str = ""
    cv_summary: str = ""  # Summary of the CV for embedding matching
    category: str = ""    # Single best matching category
    jobs: list = []       # List of jobs from the category
    filtered_jobs: list = []  # Final jobs after embedding matching

# --- GUI Layout ---
st.title("Tech Job Analyzer (LangGraph Demo)")

# Load categories at startup
JOB_CATEGORIES = load_job_categories()

# Sidebar configuration
with st.sidebar:
    st.header("API Keys & Options")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    openai_key = st.text_input("OpenAI API Key (for embeddings)", type="password")
    st.markdown("---")
    st.header("Visualization")
    show_mermaid = st.checkbox("Show Mermaid Graph Visualization")
    
    if JOB_CATEGORIES:
        st.markdown("---")
        st.header("Available Categories")
        st.write(", ".join(JOB_CATEGORIES))

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
    import logging
    logger = logging.getLogger("fetch_jobs")
    
    with st.status("Step 2: Fetching jobs from category...") as status:
        if not state.category:
            logger.warning("No category selected")
            st.error("Could not determine job category from PDF")
            return state
        
        try:            
            url = f"https://remotive.com/api/remote-jobs?category={state.category}"
            logger.info(f"Fetching jobs from category: {state.category}")
            
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            all_jobs = resp.json().get("jobs", [])
            # Get top 10 most recent jobs from category
            state.jobs = all_jobs[:10] if len(all_jobs) > 10 else all_jobs
            
            logger.info(f"Retrieved {len(state.jobs)} jobs from category {state.category}")
            status.update(label=f"✅ Found {len(state.jobs)} jobs in {state.category}!", state="complete", expanded=False)
            
        except Exception as e:
            logger.error(f"Error fetching jobs: {str(e)}")
            st.error(f"Error fetching jobs: {str(e)}")
            state.jobs = []
            
    return state

def filter_by_experience(state):
    import logging
    logger = logging.getLogger("filter_by_experience")
    
    with st.status("Step 3: Filtering jobs by semantic similarity...") as status:
        jobs = state.jobs
        logger.info(f"Filtering {len(jobs)} jobs using semantic matching")
        
        # Create embeddings for CV summary and job descriptions if OpenAI key is available
        embeddings_available = False
        if state.openai_key and state.cv_summary:
            try:
                from langchain_openai import OpenAIEmbeddings
                import numpy as np
                import os
                os.environ["OPENAI_API_KEY"] = state.openai_key
                
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                
                # Get embeddings for CV summary
                cv_embedding = embeddings.embed_query(state.cv_summary)
                
                # Get embeddings for jobs (titles + first part of descriptions)
                job_texts = [f"{job['title']} {job['description'][:500]}" for job in jobs]
                job_embeddings = embeddings.embed_documents(job_texts)
                
                embeddings_available = True
                logger.info("Successfully created embeddings for semantic matching")
            except Exception as e:
                logger.error(f"Error creating embeddings: {str(e)}")
                embeddings_available = False
        
        filtered_jobs = []
        # Set up semantic matching if OpenAI key is available
        if embeddings_available:
            try:
                # Calculate semantic similarity scores
                for idx, job in enumerate(jobs):
                    similarity = float(np.dot(cv_embedding, job_embeddings[idx]))
                    match_score = (similarity + 1) / 2  # Normalize to 0-1 range
                    job["match_score"] = match_score
                    filtered_jobs.append(job)
                    logger.debug(f"Job {job['title']}: match_score={match_score:.2f}")
                
                # Sort by match score
                filtered_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            except Exception as e:
                logger.error(f"Error calculating semantic scores: {str(e)}")
                filtered_jobs = jobs  # Fallback to all jobs if embeddings fail
        else:
            logger.warning("OpenAI key not available, returning all jobs without semantic matching")
            filtered_jobs = jobs
        
        state.filtered_jobs = filtered_jobs[:3]  # Keep top 3 matches
        
        logger.info(f"Found {len(filtered_jobs)} positions after semantic matching")
        if filtered_jobs:
            logger.info(f"Top match: {filtered_jobs[0]['title']} with score {filtered_jobs[0].get('match_score', 0):.2f}")
        
        status.update(label=f"✅ Found {len(filtered_jobs)} matching positions!", state="complete", expanded=False)
        
    return state

def embed_jobs(state):
    jobs = state.jobs
    openai_key = state.openai_key
    if not openai_key:
        state.embeddings = [None]*len(jobs)
    else:
        from langchain_openai import OpenAIEmbeddings
        import os
        os.environ["OPENAI_API_KEY"] = openai_key
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        state.embeddings = embeddings.embed_documents([j["description"][:2000] for j in jobs])
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
    import logging
    logger = logging.getLogger("analyze_pdf")
    logger.info(f"Starting PDF analysis. PDF text length: {len(state.pdf_text) if state.pdf_text else 0}")
    
    if not state.pdf_text or not state.anthropic_key:
        st.error("Please provide both a PDF and API key")
        return state
    
    with st.status("Step 1: Analyzing PDF content...") as status:
        client = Anthropic(api_key=state.anthropic_key)
        
        # Prepare the prompt
        categories_str = ", ".join(JOB_CATEGORIES)
        prompt = f"""Analyze the following CV or job description and provide exactly two things:

1. The single best matching job category from this list: {categories_str}
2. A concise summary (max 200 words) of the candidate's profile, experience, and key skills. This summary will be used for semantic matching with job descriptions.

Format the response as a valid JSON object with two fields:
- 'category': The best matching category name (must be exactly as shown in the list)
- 'summary': The profile summary

Here's the text to analyze:\n\n{state.pdf_text[:4000]}"""
        
        try:
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text
            logger.info(f"Raw LLM response: {result}")
            
            # Extract JSON from response
            import json
            import re
            
            json_str = result.strip()
            if not (json_str.startswith('{') and json_str.endswith('}')):
                match = re.search(r"\{[\s\S]*\}", result)
                if match:
                    json_str = match.group(0)
            
            parsed = json.loads(json_str)
            logger.info(f"Parsed response: {parsed}")
            
            # Update state with analysis results
            if "category" in parsed and parsed["category"] in JOB_CATEGORIES:
                state.category = parsed["category"]
            else:
                logger.warning("Invalid or missing category in response")
                state.category = JOB_CATEGORIES[0]  # Fallback to first category
                
            if "summary" in parsed:
                state.cv_summary = parsed["summary"]
            
            # Display results
            st.write("📑 Analysis Results:")
            st.write(f"**Selected Category:** {state.category}")
            st.write("**Profile Summary:**")
            st.info(state.cv_summary)
            
            status.update(label="✅ PDF Analysis Complete!", state="complete", expanded=False)
            
        except Exception as e:
            logger.error(f"Error analyzing PDF: {e}")
            st.error(f"Error analyzing PDF: {str(e)}")
            
    return state

# --- LangGraph definition ---
def make_graph():
    builder = StateGraph(state_schema=JobGraphState)
    
    # Add nodes
    builder.add_node("analyze_pdf", analyze_pdf)
    builder.add_node("fetch_jobs", fetch_jobs)
    builder.add_node("filter_jobs", filter_by_experience)
    
    # Add edges
    builder.add_edge("analyze_pdf", "fetch_jobs")
    builder.add_edge("fetch_jobs", "filter_jobs")
    builder.add_edge("filter_jobs", END)
    
    # Set entry point
    builder.set_entry_point("analyze_pdf")
    
    return builder.compile()

# Create the graph
graph = make_graph()

def plot_agent_graph():
    G = nx.DiGraph()
    # Add nodes with custom shapes and colors
    G.add_node("analyze_pdf", label="Analyze PDF\n& Match Category", shape="rounded", fillcolor="#E1BEE7")
    G.add_node("fetch_jobs", label="Fetch Jobs\nby Category", shape="rounded", fillcolor="#E3F2FD")
    G.add_node("filter_jobs", label="Filter Jobs\nby Similarity", shape="rounded", fillcolor="#E8F5E9")
    G.add_node("END", label="END", shape="rectangle", fillcolor="#ECEFF1")
    
    # Edges
    G.add_edges_from([
        ("analyze_pdf", "fetch_jobs"),
        ("fetch_jobs", "filter_jobs"),
        ("filter_jobs", "END"),
    ])
    
    # Arrange nodes vertically
    pos = {
        "analyze_pdf": (0, 3),
        "fetch_jobs": (0, 2),
        "filter_jobs": (0, 1),
        "END": (0, 0),
    }
    
    fig, ax = plt.subplots(figsize=(4, 6))
    node_colors = [G.nodes[n].get("fillcolor", "#FFFFFF") for n in G.nodes]
    node_labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5000, edgecolors="#333333", linewidths=2, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=30, edge_color="#888888", width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold", font_color="#222222", ax=ax)
    ax.set_axis_off()
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, width=400)

st.header("Agent Workflow Graph")
plot_agent_graph()

if go and anthropic_key:
    with st.spinner("Running pipeline..."):
        initial_state = JobGraphState(
            anthropic_key=anthropic_key,
            openai_key=openai_key,
            pdf_text=pdf_text or ""
        )
        
        result_state = graph.invoke(initial_state)
        jobs = result_state["filtered_jobs"]
        
        if len(jobs) == 0:
            st.warning("No matching jobs found. Try another job category or adjust your CV content.")
        else:
            st.success(f"🎯 Found the top {len(jobs)} matching positions in {result_state['category']}!")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Quick View", "📑 Detailed View"])
            
            with tab1:
                # Compact view with key information
                for i, job in enumerate(jobs, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### {i}. {job['title']}")
                            st.markdown(f"**Company:** {job['company_name']}")
                            st.markdown(f"**Location:** {job['candidate_required_location']}")
                            match_score = int(job.get('match_score', 0.5) * 100)
                            st.progress(job.get('match_score', 0.5), text=f"Match Score: {match_score}%")
                        with col2:
                            st.link_button("Apply Now", job['url'], use_container_width=True)
                    st.divider()
            
            with tab2:
                # Detailed view with full descriptions
                for i, job in enumerate(jobs, 1):
                    with st.expander(f"{i}. {job['title']} at {job['company_name']}"):
                        st.markdown("### Job Details")
                        st.markdown(f"**Location:** {job['candidate_required_location']}")
                        st.markdown(f"**Job Type:** {job.get('job_type', 'Not specified')}")
                        st.markdown(f"**Match Score:** {int(job.get('match_score', 0.5) * 100)}%")
                        
                        st.markdown("### Full Description")
                        st.markdown(job["description"])
                        
                        st.link_button("Apply for this Position", job['url'], use_container_width=True)
else:
    st.info("👋 Welcome! To get started:\n\n" + 
            "1. Enter your API keys in the sidebar\n" +
            "2. Upload your CV or a job description PDF\n" +
            "3. Click 'Analyze PDF & Search Jobs' to find matching positions")