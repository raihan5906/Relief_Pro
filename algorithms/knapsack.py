# algorithms/knapsack.py

def solve_01_knapsack(items, capacity):
    """
    Solves 0/1 Knapsack using Dynamic Programming.
    Items: List of (name, weight, value)
    """
    n = len(items)
    # dp[i][w] stores the maximum value for i items and capacity w
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Filling the DP table bottom-up
    for i in range(1, n + 1):
        name, weight, value = items[i-1]
        for w in range(capacity + 1):
            if weight <= w:
                dp[i][w] = max(value + dp[i-1][w-weight], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]

    # Backtracking logic to retrieve selected item details
    selected_items = []
    current_val = dp[n][capacity]
    current_weight = capacity

    for i in range(n, 0, -1):
        if current_val <= 0:
            break
        # If value changed, it means item 'i-1' was included
        if current_val != dp[i-1][current_weight]:
            item = items[i-1]
            selected_items.append(item)
            current_val -= item[2]
            current_weight -= item[1]

    # Returns total value, list of items, and remaining empty capacity
    return dp[n][capacity], selected_items, current_weight

def solve_fractional_knapsack(items, capacity):
    """
    Solves Fractional Knapsack using Greedy Technique.
    Items: List of (name, weight, value)
    """
    # Greedy Choice: Sort by value/weight ratio descending
    items.sort(key=lambda x: x[2]/x[1], reverse=True)

    total_value = 0
    selected_details = []

    for name, weight, value in items:
        if capacity <= 0:
            break
        if weight <= capacity:
            capacity -= weight
            total_value += value
            selected_details.append((name, weight, 1.0)) # 1.0 = 100% taken
        else:
            # Take the remaining space as a fraction of this item
            fraction = capacity / weight
            total_value += value * fraction
            selected_details.append((name, weight, fraction))
            capacity = 0

    return total_value, selected_details