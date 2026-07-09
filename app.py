# app.py
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from collections import Counter
import heapq
import re

# Core Logic Imports
from algorithms.knapsack import solve_01_knapsack
from algorithms.tsp_solver import TSPSolver

# --- INTEL LINK ENGINE (GEMINI) ---
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.warning("⚠️ CRITICAL ALARM: Satellite Link Offline. Gemini API Token Unresolved.")

st.set_page_config(page_title="RELIEF PRO", layout="wide")

# --- ALGO 1: HUFFMAN CODING (UNIT III - GREEDY COMPRESSION) ---
def get_huffman_encoding(items):
    if not items: return {}
    manifest_str = "".join([item[0] for item in items])
    if not manifest_str: return {}
    freq = Counter(manifest_str)
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo, hi = heapq.heappop(heap), heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = '0' + pair[1]
        for pair in hi[1:]: pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    res = heapq.heappop(heap)[1:]
    return {p[0]: p[1] for p in res}

# --- ENTERPRISE UI STYLING & SKELETON CSS ENGINE ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem !important; padding-bottom: 0.5rem !important; max-width: 98% !important; }
    div[data-baseweb="select"] { cursor: pointer !important; }
    .stButton>button { width: 100%; border-radius: 4px; height: 3em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .report-box { padding: 12px; border-radius: 6px; border: 1px solid #dcdfe6; background-color: #f8f9fb; margin-bottom: 12px; }
    .stage-card { text-align: center; padding: 10px; border-radius: 4px; border: 1px solid #e4e7ed; background: #fff; }
    .stage-active { border-left: 4px solid #4B9AFF !important; font-weight: bold; background: #f0f7ff; }
    .skeleton-box {
        background: linear-gradient(90deg, #f2f4f7 25%, #e4e7ed 50%, #f2f4f7 75%);
        background-size: 200% 100%;
        animation: pulse 1.4s infinite ease-in-out;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    @keyframes pulse {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ RELIEF-FORCE")
st.markdown("<p style='margin-top:-15px; color:#606266;'>Automated Resource Constraints & State-Space Deployment Node Router</p>", unsafe_allow_html=True)
st.markdown("---")

# --- TACTICAL WIZARD ROUTING VARIABLES ---
if 'active_stage' not in st.session_state: st.session_state.active_stage = 1
if 'capacity_val' not in st.session_state: st.session_state.capacity_val = 200
if 'pack_computed' not in st.session_state: st.session_state.pack_computed = False
if 'route_computed' not in st.session_state: st.session_state.route_computed = False

def sync_slider(): st.session_state.capacity_val = st.session_state.cap_input
def sync_input(): st.session_state.capacity_val = st.session_state.cap_slider

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🕹️ Deployment Parameters")
scenario = st.sidebar.selectbox("Active Threat Scenario", ["Normal Operations", "Major Flood", "Severe Earthquake", "Urban Wildfire", "Chemical Leak"])

st.sidebar.markdown("<small>**Payload Weight Capacity Threshold (kg)**</small>", unsafe_allow_html=True)
sc_1, sc_2 = st.sidebar.columns([2, 1])
sc_1.slider("Slider", 50, 1000, key="cap_slider", on_change=sync_input, label_visibility="collapsed", value=st.session_state.capacity_val)
sc_2.number_input("Input", 50, 1000, key="cap_input", on_change=sync_slider, label_visibility="collapsed", value=st.session_state.capacity_val)

num_shelters = st.sidebar.number_input("Target Delivery Nodes", min_value=3, max_value=8, value=4)
shelter_names = ["Central Hub"] + [f"Node {chr(65+i)}" for i in range(num_shelters-1)]

# --- AI DATA INGESTION ENGINE ---
def get_ai_manifest(scen):
    fallback_items = [
        ("Potable Water Core Supply", 25, 2900),
        ("Emergency Food Rations", 20, 2800),
        ("Trauma Medical Kit", 12, 2400),
        ("Satellite Comms Case", 8, 900),
        ("Emergency Shelter Tents", 45, 1200),
        ("Portable Power Generator", 55, 1500),
        ("High-Capacity Water Filter", 18, 1100),
        ("Hazardous Material Suits", 14, 850),
        ("Search & Rescue Grade Drone", 7, 950),
        ("Blankets & Thermal Gear", 22, 600),
        ("Heavy-Duty Debris Shovel", 11, 400),
        ("Inflatable Rescue Skiff", 40, 1300)
    ]
    
    if not api_key:
        return fallback_items

    prompt = f"""
    Act as a professional disaster logistics commander. For the emergency scenario '{scen}', suggest exactly 12 unique, highly critical rescue/relief tools.
    CRITICAL: You MUST include an entry explicitly named 'Potable Water Core Supply' and an entry explicitly named 'Emergency Food Rations'.
    Provide a realistic mix of masses from 5kg up to 85kg. Higher priorities should generally reflect vital life-support status.
    Return ONLY a valid Python list of tuples with no explanation text, using this format exactly: [("Name String", weight_int, value_int), ...]
    """
    try:
        response = model.generate_content(prompt)
        text_content = response.text.strip()
        text_content = re.sub(r'^```[a-zA-Z]*\n', '', text_content)
        text_content = re.sub(r'\n```$', '', text_content).strip()
        
        raw_data = eval(text_content)
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            return fallback_items
            
        boosted = []
        for n, w, v in raw_data:
            if any(x in str(n).lower() for x in ["water", "food", "ration", "medical"]):
                boosted.append((str(n), int(w), int(v) + 2500))
            else:
                boosted.append((str(n), int(w), int(v)))
        return boosted
    except Exception as e:
        return fallback_items

if 'manifest' not in st.session_state or st.session_state.get('cached_scen') != scenario:
    with st.spinner(f"Analyzing Requirements For {scenario}..."):
        st.session_state.manifest = get_ai_manifest(scenario)
        st.session_state.cached_scen = scenario

# Global Algorithm Evaluators
dp_val, picked_items, remaining = solve_01_knapsack(st.session_state.manifest, int(st.session_state.capacity_val))
picked_names = [x[0] for x in picked_items]

# --- PIPELINE BREADCRUMB PROGRESS TRACKER ---
b_col1, b_col2, b_col3, b_col4 = st.columns(4)
titles = ["Stage 1: Resource Loading", "Stage 2: Topology Optimization", "Stage 3: Cipher Encoding", "Stage 4: Deployment Sheet"]
for idx, col in enumerate([b_col1, b_col2, b_col3, b_col4]):
    active_style = "stage-card stage-active" if st.session_state.active_stage == idx + 1 else "stage-card"
    col.markdown(f"<div class='{active_style}'>{titles[idx]}</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Shared Mapping Infrastructure Function (Fixed background lines & layout visibility)
def draw_base_graph(p_matrix, active_edges=None):
    G = nx.DiGraph()
    for i in range(num_shelters):
        for j in range(num_shelters):
            if i != j: G.add_edge(shelter_names[i], shelter_names[j], weight=int(p_matrix[i][j]))
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    
    # Always draw all non-blocked background network routing lanes
    bg_edges = [e for e in G.edges if G.edges[e]['weight'] < 500]
    nx.draw_networkx_edges(G, pos, edgelist=bg_edges, edge_color='#E4E7ED', width=1.5, alpha=0.7, ax=ax)
    
    for u, v, d in G.edges(data=True):
        if d['weight'] >= 500:
            ax.text((pos[u][0]+pos[v][0])/2, (pos[u][1]+pos[v][1])/2, "🛑 CLOSED", color='red', weight='bold', fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    if active_edges:
        l_map = {}
        for m, edge in enumerate(active_edges):
            u_node, v_node = edge
            l_map[u_node] = f"{u_node}\n(Seq {m+1})"
            if m == len(active_edges) - 1:
                l_map[v_node] = f"{v_node}\n(Seq {m+2})"
                
        nx.draw_networkx_nodes(G, pos, node_color='#FFD700', node_size=1500, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=l_map, font_size=8, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edgelist=active_edges, edge_color='#4B9AFF', width=4, arrows=True, connectionstyle="arc3,rad=0.1", ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_color='#E4E7ED', node_size=1300, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8)

    labels = {(u, v): f"{d['weight']}km" for u, v, d in G.edges(data=True) if d['weight'] < 500}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
    return fig

# =========================================================================
# STAGE 1: CARGO LOADING MANAGEMENT
# =========================================================================
if st.session_state.active_stage == 1:
    st.subheader("📦 Resource Loading & Inventory Knapsack Matrices")
    
    with st.expander("🛠️ Inject Custom Deployment Asset"):
        cx1, cx2, cx3 = st.columns([2, 1, 1])
        in_name = cx1.text_input("Asset Label")
        in_wt = cx2.number_input("Mass (kg)", 1, 500, 20)
        in_val = cx3.number_input("Utility Priority", 1, 4000, 200)
        if st.button("Commit Node to Memory") and in_name:
            st.session_state.manifest.append((in_name, int(in_wt), int(in_val)))
            st.rerun()

    if st.button("Run Dynamic Programming Allocation Strategy", type="primary"):
        sk_placeholder = st.empty()
        with sk_placeholder.container():
            st.markdown("<div class='skeleton-box' style='height:35px; width:30%;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='skeleton-box' style='height:200px; width:100%;'></div>", unsafe_allow_html=True)
        time.sleep(1.2)
        st.session_state.pack_computed = True
        sk_placeholder.empty()

    if st.session_state.pack_computed:
        m1, m2 = st.columns(2)
        m1.metric("Calculated Value Matrix Output", dp_val)
        m2.metric("Payload Mass Consumed", f"{int(st.session_state.capacity_val)-remaining} / {int(st.session_state.capacity_val)} kg")
        
        records = []
        for name, w, v in sorted(st.session_state.manifest, key=lambda x: x[0] in picked_names, reverse=True):
            status = "✅ Selected" if name in picked_names else "❌ Rejected"
            reason = "Optimal Optimization Node" if name in picked_names else ("Exceeds Limit Value" if w > int(st.session_state.capacity_val) else "Sub-Optimal Density Match")
            records.append({"Equipment Profiling": name, "Mass (kg)": w, "Decision State": status, "Algorithmic Reason": reason})
        
        st.dataframe(pd.DataFrame(records), height=280, use_container_width=True)
        if st.button("Lock Cargo and Proceed to Routing ➔"):
            st.session_state.active_stage = 2
            st.rerun()

# =========================================================================
# STAGE 2: GEOGRAPHIC ROUTE PLANNING (SMOOTH ANOMATION & BASE ROUTE NETWORK)
# =========================================================================
elif st.session_state.active_stage == 2:
    st.subheader("📍 Topology Optimization via Branch and Bound")
    
    if 'net_matrix' not in st.session_state or st.session_state.get('last_num_s') != num_shelters:
        mat = np.random.randint(15, 60, size=(num_shelters, num_shelters))
        np.fill_diagonal(mat, 0)
        st.session_state.net_matrix = (mat + mat.T) // 2
        st.session_state.last_num_s = num_shelters

    p_matrix = np.copy(st.session_state.net_matrix)
    if scenario == "Severe Earthquake" and num_shelters > 2: p_matrix[0, 2] = p_matrix[2, 0] = 999

    compute_route_btn = st.button("Compute State Space Optimal Network Path", type="primary")
    
    map_viewport = st.empty()

    if compute_route_btn:
        solver = TSPSolver(p_matrix.tolist())
        min_cost, best_path = solver.solve()
        st.session_state.tsp_cost = min_cost
        st.session_state.tsp_path = best_path
        st.session_state.route_computed = True
        
        animated_edges = []
        for step in range(len(best_path) - 1):
            u_node = shelter_names[best_path[step]]
            v_node = shelter_names[best_path[step + 1]]
            animated_edges.append((u_node, v_node))
            
            with map_viewport.container():
                st.pyplot(draw_base_graph(p_matrix, active_edges=animated_edges))
            time.sleep(0.3)  # Speed adjusted to 0.3s for smooth traversal animation
            plt.close()
    else:
        with map_viewport.container():
            if st.session_state.route_computed:
                final_edges = [(shelter_names[st.session_state.tsp_path[k]], shelter_names[st.session_state.tsp_path[k+1]]) for k in range(len(st.session_state.tsp_path)-1)]
                st.pyplot(draw_base_graph(p_matrix, active_edges=final_edges))
            else:
                st.pyplot(draw_base_graph(p_matrix))
            plt.close()

    st.markdown("---")
    if not st.session_state.route_computed:
        st.info("Awaiting Algorithmic Traversal Execution.")
    else:
        st.success(f"Optimized Minimal Path Verified: {st.session_state.tsp_cost} km")
        solver = TSPSolver(p_matrix.tolist())
        solver.solve()
        routes_df = pd.DataFrame([{"Overhead (km)": c, "Sequence Trace": " ➔ ".join([shelter_names[i] for i in p])} for c, p in solver.all_routes])
        st.dataframe(routes_df.drop_duplicates('Sequence Trace').sort_values('Overhead (km)'), height=180, use_container_width=True)

        if st.button("Freeze Path Metrics and Proceed to Telemetry Ciphering ➔"):
            st.session_state.active_stage = 3
            st.rerun()

# =========================================================================
# STAGE 3: TELEMETRY COMPRESSION
# =========================================================================
elif st.session_state.active_stage == 3:
    st.subheader("🔐 Comms Compression via Huffman Prefix Trees")
    st.info("Greedy character allocation framework optimizes cargo text data packages into prefix-free bit representations for degraded satellite frequencies.")
    
    if not picked_items:
        st.warning("No tracking data available. Return to Stage 1.")
    else:
        huff_codes = get_huffman_encoding(picked_items)
        comp_records = []
        for item in picked_items:
            name = item[0]
            b_orig = len(name) * 8
            b_comp = sum(len(huff_codes.get(c, '')) for c in name)
            comp_records.append({
                "Cargo String Profile": name,
                "Native Size (ASCII)": f"{b_orig} bits",
                "Optimized Size (Huffman)": f"{b_comp} bits",
                "Bandwidth Savings": f"{((b_orig-b_comp)/b_orig)*100:.1f}%",
                "Assigned Prefix Pipeline": "".join([huff_codes.get(c, '') for c in name])
            })
        st.dataframe(pd.DataFrame(comp_records), height=250, use_container_width=True)
        
        if st.button("Generate Consolidated Mission Document ➔"):
            st.session_state.active_stage = 4
            st.rerun()

# =========================================================================
# STAGE 4: MASTER DISPATCH DOCUMENT
# =========================================================================
elif st.session_state.active_stage == 4:
    st.subheader("📋 Master Mission Manifest & Telemetry Blueprint")
    
    # Block 1: Allocation Profile
    st.markdown("<div class='report-box'><strong>📦 Allocation Profile & Packing Metrics</strong></div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Selected Item": n, "Mass": f"{w} kg"} for n, w, _ in picked_items]), height=150, use_container_width=True)
    st.metric("Total Strategic Value Metric", dp_val)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Block 2: Routing Path Configuration & Full Topology Map View
    st.markdown("<div class='report-box'><strong>📍 Routing Execution Order & Cost Constraints</strong></div>", unsafe_allow_html=True)
    route_str = " ➔ ".join([shelter_names[idx] for idx in st.session_state.get('tsp_path', [])])
    st.info(f"**Optimal Node Chain Sequence:** {route_str}")
    st.metric("Total Structural Traversal Overhead", f"{st.session_state.get('tsp_cost', 0)} km")
    
    p_mat_final = np.copy(st.session_state.net_matrix)
    if scenario == "Severe Earthquake" and num_shelters > 2: p_mat_final[0, 2] = p_mat_final[2, 0] = 999
    
    final_edges_trace = [(shelter_names[st.session_state.tsp_path[k]], shelter_names[st.session_state.tsp_path[k+1]]) for k in range(len(st.session_state.tsp_path)-1)]
    
    sc_col1, sc_col2, sc_col3 = st.columns([1, 4, 1])
    with sc_col2:
        # Includes background topology grid lines for full summary sheet printing
        st.pyplot(draw_base_graph(p_mat_final, active_edges=final_edges_trace))
        plt.close()
    st.markdown("<br>", unsafe_allow_html=True)

    # Block 3: Binary Streams Matrix
    st.markdown("<div class='report-box'><strong>🔐 Encoded Manifest Streams (Compressed Binary Footprint)</strong></div>", unsafe_allow_html=True)
    h_codes = get_huffman_encoding(picked_items)
    stream_data = []
    for item in picked_items:
        bits = "".join([h_codes.get(c, '') for c in item[0]])
        stream_data.append({"Selected Item Asset": item[0], "Huffman Ciphertext Token": bits})
    st.dataframe(pd.DataFrame(stream_data), height=150, use_container_width=True)

    st.markdown("---")
    if st.button("🔄 Flush Memory State and Restart Optimization Routine"):
        st.session_state.active_stage = 1
        st.session_state.pack_computed = False
        st.session_state.route_computed = False
        st.rerun()

st.caption("Strategic Infrastructure Engine // Running: Huffman Coding, Dijkstra Data Weighting, 0/1 Knapsack, TSP Branch & Bound")