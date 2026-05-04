# algorithms/tsp_solver.py
import sys

class TSPSolver:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.n = len(adj_matrix)
        self.visited = [False] * self.n
        self.final_path = [0] * (self.n + 1)
        self.final_res = sys.maxsize
        self.all_routes = []  # Stores (distance, path) for all valid circuits

    def copy_to_final(self, curr_path):
        for i in range(self.n):
            self.final_path[i] = curr_path[i]
        self.final_path[self.n] = curr_path[0]

    def first_min(self, i):
        res = sys.maxsize
        for k in range(self.n):
            if self.adj_matrix[i][k] < res and i != k:
                res = self.adj_matrix[i][k]
        return res

    def second_min(self, i):
        first, second = sys.maxsize, sys.maxsize
        for j in range(self.n):
            if i == j: continue
            if self.adj_matrix[i][j] <= first:
                second = first
                first = self.adj_matrix[i][j]
            elif self.adj_matrix[i][j] <= second:
                second = self.adj_matrix[i][j]
        return second

    def tsp_recursive(self, curr_bound, curr_weight, level, curr_path):
        if level == self.n:
            if self.adj_matrix[curr_path[level - 1]][curr_path[0]] != 0:
                curr_res = curr_weight + self.adj_matrix[curr_path[level - 1]][curr_path[0]]
                
                # Record every complete circuit found
                full_circuit = curr_path[:] + [curr_path[0]]
                self.all_routes.append((curr_res, full_circuit))
                
                if curr_res < self.final_res:
                    self.copy_to_final(curr_path)
                    self.final_res = curr_res
            return

        for i in range(self.n):
            if self.adj_matrix[curr_path[level - 1]][i] != 0 and not self.visited[i]:
                temp = curr_bound
                curr_weight += self.adj_matrix[curr_path[level - 1]][i]

                if level == 1:
                    curr_bound -= ((self.first_min(curr_path[level - 1]) + self.first_min(i)) / 2)
                else:
                    curr_bound -= ((self.second_min(curr_path[level - 1]) + self.first_min(i)) / 2)

                if curr_bound + curr_weight < self.final_res:
                    curr_path[level] = i
                    self.visited[i] = True
                    self.tsp_recursive(curr_bound, curr_weight, level + 1, curr_path)

                # Backtrack
                curr_weight -= self.adj_matrix[curr_path[level - 1]][i]
                curr_bound = temp
                self.visited = [False] * len(self.visited)
                for j in range(level):
                    self.visited[curr_path[j]] = True

    def solve(self):
        self.all_routes = [] # Reset for fresh run
        curr_path = [-1] * (self.n + 1)
        curr_bound = 0
        for i in range(self.n):
            curr_bound += (self.first_min(i) + self.second_min(i))
        curr_bound = (curr_bound + 1) // 2
        self.visited[0] = True
        curr_path[0] = 0
        self.tsp_recursive(curr_bound, 0, 1, curr_path)
        return self.final_res, self.final_path