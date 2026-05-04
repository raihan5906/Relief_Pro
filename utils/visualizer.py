# utils/visualizer.py
import matplotlib.pyplot as plt
import networkx as nx

def draw_delivery_route(path_names, adj_matrix, names):
    G = nx.DiGraph()
    # Add nodes
    for name in names:
        G.add_node(name)
    
    # Add edges for the circuit
    for i in range(len(path_names) - 1):
        G.add_edge(path_names[i], path_names[i+1])
        
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_weight='bold', arrows=True)
    plt.title("Optimized Relief Delivery Circuit")
    plt.show()