import math

def distance(a, b):
    x0, y0 = a
    x1, y1 = b
    return math.sqrt((x0-x1)**2 + (y0-y1)**2)


def energy_per_turn(grid, lighthouse_positions):
    RDIST = 5
    MAX_ENERGY = 100

    energy = []
    for y in range(len(grid)):
        energy.append([])
        for x in range(len(grid[0])):
            if grid[y][x] == 1:
                next_energy = 0
                for lh_pos in lighthouse_positions:
                    dist = distance(lh_pos, (x, y))
                    delta = int(math.floor(RDIST - dist))
                    if delta > 0:
                        next_energy += delta
                energy[y].append(min(next_energy, MAX_ENERGY))
            else:
                energy[y].append(0)
    return energy


