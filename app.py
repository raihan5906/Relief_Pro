# app.py
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from algorithms.knapsack import solve_01_knapsack
from algorithms.tsp_solver import TSPSolver

# --- AI CONFIGURATION ---
# Accessing API key from environment variable (set via terminal)
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.warning("⚠️ GEMINI_API_KEY not found in environment. Using local fallback logic.")

st.set_page_config(page_title="Relief-Force AI Optimizer", layout="wide")

# --- CSS INJECTION ---
st.markdown("""
    <style>
    div[data-baseweb="select"] { cursor: pointer !important; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .blurred-container { filter: blur(4px) grayscale(80%); opacity: 0.4; pointer-events: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚚 Relief-Force: AI-Driven Logistics Engine")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("🌍 Global Configuration")
scenario = st.sidebar.selectbox(
    "Select Disaster Scenario",
    ["Normal Operations", "Major Flood", "Severe Earthquake", "Urban Wildfire", "Chemical Leak"],
    key="disaster_selector"
)

# Capacity slider - value is passed to int() later for Knapsack safety
capacity = st.sidebar.slider("Max Vehicle Capacity (kg)", 10, 500, 100)
num_shelters = st.sidebar.number_input("Number of Shelters", min_value=3, max_value=8, value=4)
shelter_names = ["Warehouse"] + [f"Shelter {chr(65+i)}" for i in range(num_shelters-1)]

# --- AI ITEM SELECTION LOGIC ---
def get_ai_items(scenario_type):
    if not api_key:
        return [("Generator", 40, 100), ("Med-Kit", 15, 75), ("Tent", 20, 80)]
    
    prompt = f"""
    Act as a disaster relief expert. For the scenario '{scenario_type}', suggest 5 essential relief items.
    Return ONLY a Python list of tuples in the format: [("ItemName", Weight_int, SurvivalValue_int)]
    Ensure weights and values are integers. Do not provide any text other than the list.
    """
    try:
        response = model.generate_content(prompt)
        # Safely evaluate string to list and cast weights to int to prevent Float indexing errors
        raw_items = eval(response.text.strip())
        items = [(str(name), int(weight), int(val)) for name, weight, val in raw_items]
        return items
    except Exception as e:
        return [("Emergency Water", 10, 100), ("First Aid", 5, 150)]

def adjust_map_for_scenario(matrix, scenario_type):
    mod_matrix = np.copy(matrix)
    if scenario_type == "Major Flood":
        return mod_matrix * 2
    elif scenario_type == "Severe Earthquake":
        for _ in range(2):
            i, j = np.random.randint(0, len(mod_matrix), 2)
            if i != j: mod_matrix[i][j] = mod_matrix[j][i] = 999 
    return mod_matrix

# --- SESSION STATE MANAGEMENT ---
if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = scenario
    st.session_state.custom_items = get_ai_items(scenario)

if st.session_state.current_scenario != scenario:
    with st.spinner("AI is analyzing disaster requirements..."):
        st.session_state.custom_items = get_ai_items(scenario)
    st.session_state.current_scenario = scenario

if 'custom_items' not in st.session_state:
    st.session_state.custom_items = get_ai_items(scenario)

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📦 AI-Prioritized Inventory")
    
    with st.expander("➕ Add Custom Relief Item"):
        c1, c2, c3 = st.columns(3)
        new_name = c1.text_input("Name", placeholder="e.g. Lifeboat")
        # Ensure weight and value are strictly integers
        new_wt = c2.number_input("Kg", min_value=1, step=1, placeholder="30")
        new_val = c3.number_input("Survival Val", min_value=1, step=1, placeholder="250")
        
    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("📥 Add to Pool"):
        if new_name:
            # Explicit cast to int for knapsack indexing safety
            st.session_state.custom_items.append((str(new_name), int(new_wt), int(new_val)))
            st.rerun()
        else:
            st.error("Enter item name.")

    if btn_col2.button("🔄 AI Re-Evaluate"):
        st.session_state.custom_items = get_ai_items(scenario)
        st.rerun()

    run_knapsack = st.button("📦 Pack Relief Supplies", type="primary")
    knapsack_holder = st.empty()
    
    # CRITICAL FIX: Ensure capacity is cast to int for solve_01_knapsack indexing
    dp_val, picked_atomic, remaining = solve_01_knapsack(st.session_state.custom_items, int(capacity))

    if run_knapsack:
        for item in picked_atomic:
            y_pos = np.linspace(0.85, 0.3, 12)
            for y in y_pos:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.add_patch(plt.Rectangle((0.15, 0.05), 0.7, 0.35, color='gray', alpha=0.2, linewidth=2, edgecolor='black'))
                ax.text(0.5, 0.15, "TRUCK LOADING...", ha='center', fontweight='bold', fontsize=12)
                circle = plt.Circle((0.5, y), 0.18, color='#4B9AFF', zorder=3)
                ax.add_patch(circle)
                ax.text(0.5, y, item[0], ha='center', va='center', color='white', fontsize=11, fontweight='bold')
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
                knapsack_holder.pyplot(fig)
                time.sleep(0.04)
                plt.close(fig)
        knapsack_holder.empty()
        st.success("AI-Selected items packed!")
        
    st.markdown("### Packing Report")
    st.metric("Total Survival Value", dp_val)
    st.write(f"Utilization: **{int(capacity) - remaining}/{int(capacity)} kg**")
    for item in picked_atomic:
        st.info(f"✅ {item[0]} | W: {item[1]}kg | V: {item[2]}")

with col2:
    st.subheader("📍 Interactive Routing Control")
    
    if 'base_matrix' not in st.session_state or st.session_state.get('num_s') != num_shelters:
        mat = np.random.randint(15, 50, size=(num_shelters, num_shelters))
        np.fill_diagonal(mat, 0)
        st.session_state.base_matrix = (mat + mat.T) // 2
        st.session_state.num_s = num_shelters

    current_adj_matrix = adjust_map_for_scenario(st.session_state.base_matrix, scenario)
    run_optimizer = st.button("🚀 Execute Delivery Circuit", type="primary")
    status_text = st.empty()
    chart_holder = st.empty()

    def draw_graph(route_edges, visited_labels, is_placeholder=False):
        G = nx.DiGraph()
        for i in range(num_shelters):
            for j in range(num_shelters):
                if i != j: G.add_edge(shelter_names[i], shelter_names[j], weight=int(current_adj_matrix[i][j]))
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw_networkx_nodes(G, pos, node_color='#EEE', node_size=1500, ax=ax)
        edge_labels = {(u, v): f"{d['weight']}km" for u, v, d in G.edges(data=True) if d['weight'] < 500}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='black', font_weight='bold', ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=[e for e in G.edges() if G.edges[e]['weight'] < 500], edge_color='#DDD', alpha=0.3, ax=ax)
        blocked_edges = [(u_e, v_e) for u_e, v_e, d in G.edges(data=True) if d['weight'] >= 500]
        for u_e, v_e in blocked_edges:
            x, y = (pos[u_e][0] + pos[v_e][0]) / 2, (pos[u_e][1] + pos[v_e][1]) / 2
            ax.text(x, y, "🛑 CLOSED", color='white', weight='bold', fontsize=8, ha='center', bbox=dict(facecolor='red', alpha=0.7))
        if not is_placeholder:
            nx.draw_networkx_nodes(G, pos, nodelist=list(visited_labels.keys()), node_color='#FFD700', node_size=1600, ax=ax)
            current_labels = {n: n for n in shelter_names}; current_labels.update(visited_labels)
            nx.draw_networkx_labels(G, pos, labels=current_labels, font_size=8, font_weight='bold')
            nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='#4B9AFF', width=4, ax=ax, arrows=True, arrowstyle='->', arrowsize=25, connectionstyle="arc3,rad=0.1")
        else:
            nx.draw_networkx_labels(G, pos, labels={n:n for n in shelter_names}, font_size=8)
        return fig

    if not run_optimizer:
        st.markdown('<div class="blurred-container">', unsafe_allow_html=True)
        chart_holder.pyplot(draw_graph([], {}, is_placeholder=True))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        solver = TSPSolver(current_adj_matrix.tolist())
        min_cost, best_path = solver.solve()
        route_edges = []; visited_labels = {}
        for i in range(len(best_path) - 1):
            u_idx, v_idx = best_path[i], best_path[i+1]
            u, v = shelter_names[u_idx], shelter_names[v_idx]
            route_edges.append((u, v))
            visited_labels[u] = f"{u}\n(#{i+1})"
            fig = draw_graph(route_edges, visited_labels, is_placeholder=False)
            chart_holder.pyplot(fig)
            status_text.warning(f"🚛 Routing: Step {i+1}")
            time.sleep(0.8); plt.close(fig)
        st.success(f"✅ Route Found: {min_cost} km")
        unique_routes = [{"Distance (km)": c, "Route": " ➔ ".join([shelter_names[i] for i in p])} for c, p in solver.all_routes]
        df = pd.DataFrame(unique_routes).drop_duplicates(subset=['Route']).sort_values(by="Distance (km)")
        st.dataframe(df.style.highlight_min(axis=0, subset=['Distance (km)'], color='#D4EDDA'), use_container_width=True)

st.caption("RVCE DAA | AI-Driven Optimization Build | Fixed Integer Indexing")