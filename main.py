# main.py
from algorithms.knapsack import solve_01_knapsack, solve_fractional_knapsack
from algorithms.pathfinding import dijkstra
from algorithms.tsp_solver import TSPSolver
from utils.visualizer import draw_delivery_route

def run_relief_system():
    print("=== RELIEF-FORCE OPTIMIZER STARTING ===\n")

    # --- PART 1: PACKING (Unit IV) ---
    # Goal: Maximize survival value within vehicle weight constraints
    total_capacity = 60 # Total vehicle weight limit in kg
    
    # Atomic items (cannot be split) handled by 0/1 Knapsack (DP)
    atomic_pool = [
        ("Generator", 40, 100),
        ("Med-Kit", 15, 75),
        ("Tent", 20, 80),
        ("Sat-Phone", 5, 50)
    ]
    
    # Bulk items (can be split) handled by Fractional Knapsack (Greedy)
    bulk_pool = [
        ("Water", 25, 125),
        ("Rice", 20, 60),
        ("Fuel", 10, 80)
    ]

    # Execute Packing Engine
    dp_val, atomic_picked, remaining = solve_01_knapsack(atomic_pool, total_capacity)
    greedy_val, bulk_picked = solve_fractional_knapsack(bulk_pool, remaining)

    print(f"[PACKING REPORT]")
    print(f"Total Value Secured: {dp_val + greedy_val}")
    print(f"Atomic Items Packed: {[x[0] for x in atomic_picked]}")
    # bulk_picked returns (name, weight, fraction); we display fraction as percentage
    print(f"Bulk Items Packed: {[(x[0], f'{int(x[2]*100)}%') for x in bulk_picked]}")
    print(f"Remaining Capacity: {remaining}kg\n")

    # --- PART 2: ROUTING (Unit IV) ---
    # Goal: Find the single shortest path to a high-priority destination
    city_map = {
        'Warehouse': {'Shelter_A': 10, 'Shelter_B': 20},
        'Shelter_A': {'Warehouse': 10, 'Shelter_C': 15, 'Shelter_D': 30},
        'Shelter_B': {'Warehouse': 20, 'Shelter_D': 10},
        'Shelter_C': {'Shelter_A': 15, 'Shelter_D': 5},
        'Shelter_D': {'Shelter_B': 10, 'Shelter_C': 5, 'Shelter_A': 30}
    }

    dist, path = dijkstra(city_map, 'Warehouse', 'Shelter_D')
    
    print(f"[ROUTING REPORT]")
    print(f"Shortest Path (Warehouse to Shelter_D): {' -> '.join(path)}")
    print(f"Total Distance: {dist} km\n")

    # --- PART 3: CIRCUIT OPTIMIZATION (Unit V) ---
    # Goal: Visit all assigned shelters and return to Warehouse optimally
    # Adjacency matrix for Shelters: 0:Warehouse, 1:Shelter_A, 2:Shelter_B, 3:Shelter_C
    adj_matrix = [
        [0, 10, 20, 25], # Warehouse
        [10, 0, 35, 15], # Shelter_A
        [20, 35, 0, 10], # Shelter_B
        [25, 15, 10, 0]  # Shelter_C
    ]
    node_names = ["Warehouse", "Shelter_A", "Shelter_B", "Shelter_C"]

    # Solve TSP using Branch and Bound
    tsp_engine = TSPSolver(adj_matrix)
    min_cost, best_path_indices = tsp_engine.solve()
    
    # Map indices back to names
    optimal_sequence = [node_names[i] for i in best_path_indices]

    print(f"[CIRCUIT OPTIMIZATION REPORT]")
    print(f"Optimal Visit Sequence: {' -> '.join(optimal_sequence)}")
    print(f"Total Circuit Distance: {min_cost} km")

    # --- PART 4: VISUALIZATION ---
    # This will open a window showing the optimized delivery route
    print("\nGenerating Route Visualization...")
    draw_delivery_route(optimal_sequence, adj_matrix, node_names)

if __name__ == "__main__":
    run_relief_system()