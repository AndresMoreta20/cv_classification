import streamlit as st
import requests
from langchain.embeddings import OpenAIEmbeddings
from anthropic import Anthropic
from langgraph.graph import StateGraph
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
import networkx as nx
import matplotlib.pyplot as plt

class JobGraphState(BaseModel):
    term: str
    experience: str
    anthropic_key: str
    openai_key: str
    jobs: list = []
    embeddings: list = []
    style: list = []


st.set_page_config(page_title="Tech Job Analyzer", layout="wide")
st.title("Tech Job Analyzer (LangGraph Demo)")

with st.sidebar:
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    openai_key = st.text_input("OpenAI API Key (for embeddings)", type="password")
    experience_level = st.selectbox("Filter by experience", ["all", "junior", "mid", "senior"])
    show_mermaid = st.checkbox("Show Mermaid Graph Visualization")

search_term = st.text_input("Job search term", "Python Developer")
go = st.button("Search & Analyze")

def fetch_jobs(state):
    term = state.term
    url = f"https://remotive.com/api/remote-jobs?search={term}"
    resp = requests.get(url, timeout=10)
    state.jobs = resp.json().get("jobs", [])
    return state

def filter_by_experience(state):
    jobs = state.jobs
    level = state.experience
    if level == "all":
        state.jobs = jobs
    else:
        level = level.lower()
        state.jobs = [j for j in jobs if level in j["title"].lower() or level in j["description"].lower()]
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

# --- LangGraph definition ---
def make_graph():
    builder = StateGraph(state_schema=JobGraphState)
    builder.add_node("fetch_jobs", fetch_jobs)
    builder.add_node("filter_by_experience", filter_by_experience)
    builder.add_node("embed_jobs", embed_jobs)
    builder.add_node("analyze_style", analyze_style)
    builder.add_edge("fetch_jobs", "filter_by_experience")
    builder.add_edge("filter_by_experience", "embed_jobs")
    builder.add_edge("embed_jobs", "analyze_style")
    builder.set_entry_point("fetch_jobs")
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
    st.pyplot(fig)

st.header("Agent Workflow Graph")
plot_agent_graph()

if go and anthropic_key:
    with st.spinner("Running pipeline..."):
        initial_state = {
            "term": search_term,
            "experience": experience_level,
            "anthropic_key": anthropic_key,
            "openai_key": openai_key,
            "jobs": [],
            "embeddings": [],
            "style": [],
        }
        result_state = graph.invoke(initial_state)
        jobs = result_state["jobs"]
        embeddings = result_state["embeddings"]
        style = result_state["style"]
        for idx, job in enumerate(jobs):
            job["embedding"] = embeddings[idx] if embeddings else None
            job["style_analysis"] = style[idx]
    st.success(f"Found {len(jobs)} jobs after filtering/analyzing")
    for job in jobs[:10]:
        st.markdown(f"**{job['title']}** at {job['company_name']}  \n"
                    f"{job['candidate_required_location']}  ")
        st.markdown(f"_{job['style_analysis']}_")
        with st.expander("View full job description"):
            st.write(job["description"])
else:
    st.info("Enter your API key and search a job to start.")