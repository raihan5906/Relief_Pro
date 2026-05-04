# algorithms/pathfinding.py
import heapq

def dijkstra(graph, start_node, end_node):
    """
    Finds the shortest path between start and end nodes.
    graph: Dictionary { node: {neighbor: weight} }
    """
    # Priority Queue to store (distance, current_node)
    pq = [(0, start_node)]
    distances = {node: float('infinity') for node in graph}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph}

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == end_node:
            break

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    # Reconstruct the path from end to start
    path = []
    curr = end_node
    while curr is not None:
        path.append(curr)
        curr = previous_nodes[curr]
    
    return distances[end_node], path[::-1] # Reverse to get start -> end