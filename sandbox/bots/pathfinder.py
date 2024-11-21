from collections import deque
from random import shuffle


def is_valid(x, y, grid, visited):
    """
    Check if the cell (x, y) is within bounds, is soil (1), and not yet visited.
    """
    rows, cols = len(grid), len(grid[0])
    return 0 <= x < cols and 0 <= y < rows and grid[y][x] == 1 and not visited[y][x]


def find_path(grid: list[list[int]],
              energy_per_turn_grid: list[list[int]],
              current_energy_grid: list[list[int]],
              start: tuple[int, int],
              end: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    if not grid or not grid[0]:
        return [], 0

    # Directions for vertical, horizontal, and diagonal movements
    directions = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
    shuffle(directions)

    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    # Queue for BFS: stores (current_position, path_to_current_position)
    queue = deque([(start, [start], energy_per_turn_grid[start[1]][start[0]])])
    visited[start[1]][start[0]] = True

    # Variables to track the best path and reward at the shortest length
    best_path = []
    max_reward = float('-inf')

    while queue:
        (current_x, current_y), path, reward = queue.popleft()

        # If we reach the end, return the path
        if (current_x, current_y) == end:
            if len(path) == len(best_path):
                # Update only if the reward is higher for the same path length
                if reward > max_reward:
                    best_path = path
                    max_reward = reward
            elif not best_path or len(path) < len(best_path):
                # Update if this is a shorter path
                best_path = path
                max_reward = reward
            continue

        # Explore neighbors
        for dx, dy in directions:
            new_x, new_y = current_x + dx, current_y + dy

            if is_valid(new_x, new_y, grid, visited):
                visited[new_y][new_x] = True
                if current_energy_grid[new_y][new_x]:
                    new_reward = reward + min(100,
                                              current_energy_grid[new_y][new_x] + energy_per_turn_grid[new_y][new_x] * (
                                                      len(path) - 1))
                else:
                    new_reward = reward + min(100, energy_per_turn_grid[new_y][new_x] * (len(path) - 1))
                queue.append(((new_x, new_y), path + [(new_x, new_y)], new_reward))

    # If no path is found, return an empty list
    return best_path, max_reward
