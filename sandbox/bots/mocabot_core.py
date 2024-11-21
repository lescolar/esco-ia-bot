#!/usr/bin/python
# -*- coding: utf-8 -*-
# Adapted from https://github.com/marcan/lighthouses_aicontest

from bots import pathfinder
from bots.geom import intersect, colinear
from bots.map_utils import energy_per_turn

class Me:

    def __init__(self, player_num: int, coor: tuple[int, int], energy: int, view: list[list[int]]):
        self.player_num = player_num
        self.coor = coor
        self.energy = energy
        self.view = view


class LightHouse:

    def __init__(self, coor: tuple[int, int], energy: int, connections: list[list[int]], owner: int, have_key: bool):
        self.coor = coor
        self.energy = energy
        self.connections = connections
        self.owner = owner
        self.have_key = have_key


class Turn:

    def __init__(self, game_map: list[list[int]], me: Me, lhs: list[LightHouse]):
        self.game_map = game_map
        self.me = me
        self.lhs = lhs

    def select_action(self) -> tuple[str, any]:
        cx, cy = self.me.coor
        lighthouses: dict[tuple[int, int], LightHouse] = {}
        for lh in self.lhs:
            lighthouses[lh.coor] = lh

        energy_per_turn_grid = energy_per_turn(self.game_map, lighthouses.keys())
        current_energy_grid = [[-1] * len(energy_per_turn_grid[0]) for _ in range(len(energy_per_turn_grid))]

        for y in range(len(energy_per_turn_grid)):
            for x in range(len(energy_per_turn_grid[0])):
                if abs(y - cy) <= 3 and abs(x - cx) <= 3:
                    current_energy_grid[y][x] = self.me.view[y - cy + 3][x - cx + 3]

        lh_paths = LightHousePaths.compute_connections(self.game_map,
                                                       energy_per_turn_grid,
                                                       current_energy_grid,
                                                       (cx, cy),
                                                       self.lhs)

        connections = Connections.compute_connections(self.lhs, self.me.player_num)

        # If there is a lighthouse in the current position
        if (cx, cy) in lighthouses.keys():
            # Probability 60%: connect to valid remote lighthouse
            if lighthouses[(cx, cy)].owner == self.me.player_num and lighthouses[(cx, cy)].energy > 10:
                for dest in lighthouses.keys():
                    if (
                            dest != (cx, cy)
                            and lighthouses[dest].have_key
                            and (cx, cy) not in lighthouses[dest].connections
                            and lighthouses[dest].owner == self.me.player_num
                            and connections.is_valid(((cx, cy), dest))
                    ):
                        print(f"CONECTANDO desde {(cx, cy)} con el faro {dest}")
                        return "connect", dest
            elif self.me.energy > lighthouses[(cx, cy)].energy:
                print(f"ATACANDO faro {(cx, cy)} con energía {lighthouses[(cx, cy)].energy}")
                return "attack", self.me.energy

        # Make triangles
        owned_lhs = [lh for lh in self.lhs if lh.owner == self.me.player_num]
        if len(owned_lhs) >= 3:
            candidate_tiles = connections.search_for_closable_lh_tile()
            if len(candidate_tiles) > 0:
                nearest_tile = lh_paths.get_nearest_tile(candidate_tiles)
                if nearest_tile == (cx, cy):
                    print(f"Esperando en {(cx, cy)} la llave")
                    return "wait", ""
                else:
                    move = lh_paths.move_towards(nearest_tile)
                    print(f"YENDO desde {(cx, cy)} a faro {nearest_tile} para cerrar")
                    return "move", move

        # Move to an unowned
        unowned_lhs = [lh for lh in self.lhs if (lh.owner != self.me.player_num and self.me.energy >= lh.energy)]
        if len(unowned_lhs) == 0:
            directions = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
            tile_energy = energy_per_turn_grid[cx][cy]
            best_direction = (0, 0)
            for dx, dy in directions:
                if current_energy_grid[cx + dx][cy + dy] > tile_energy:
                    tile_energy = current_energy_grid[cx + dx][cy + dy]
                    best_direction = (dx, dy)
            if best_direction == (0, 0):
                print(f"Esperando en {(cx, cy)} la llave")
                return "wait", ""
            else:
                print(f"BUSCANDO faro desde {(cx, cy)} con {best_direction} ")
                return "move", best_direction
        else:
            paths = [lh_paths.paths_dict[lh.coor] for lh in unowned_lhs]
            energetic_paths = [p for (p, r) in paths if len(p) * 10 < (self.me.energy + r)]
            if len(energetic_paths) > 0:
                shortest_path = sorted(energetic_paths, key=lambda p: len(p))[0]
            else:
                shortest_path = sorted([p for (p, r) in paths], key=lambda p: len(p))[0]
            # Check if the move is valid
            move = (shortest_path[1][0] - shortest_path[0][0], shortest_path[1][1] - shortest_path[0][1])
            return "move", move


class LightHousePaths:
    def __init__(self, paths_dict):
        self.paths_dict = paths_dict

    def get_nearest_tile(self, lh_tiles):
        return sorted(lh_tiles, key=lambda tile: self.get_path_len(tile))[0]

    def get_path_len(self, destiny):
        return len(self.paths_dict[destiny][0])

    def move_towards(self, destiny):
        (computed_path, _) = self.paths_dict[destiny]
        return computed_path[1][0] - computed_path[0][0], computed_path[1][1] - computed_path[0][1]

    @staticmethod
    def compute_connections(grid: list[list[int]],
                            energy_per_turn_grid: list[list[int]],
                            current_energy_grid: list[list[int]],
                            origin: tuple[int, int],
                            lighthouses: list[LightHouse]):
        lh_paths = dict(
            (tuple(lh.coor),
             pathfinder.find_path(grid, energy_per_turn_grid, current_energy_grid, origin, lh.coor))
            for lh in lighthouses)
        return LightHousePaths(lh_paths)


class Connections:
    def __init__(self, current, missing_with_origin_key, missing_with_destiny_key, missing_without_key):
        self.current = current
        self.missing_with_origin_key = missing_with_origin_key
        self.missing_with_destiny_key = missing_with_destiny_key
        self.missing_without_key = missing_without_key

    def is_valid(self, candidate):
        return all(not intersect(candidate, current) for current in self.current)

    def search_for_closable_lh_tile(self):
        candidates = []
        for candidate in self.missing_with_origin_key:
            if all(not intersect(candidate, current) for current in self.current):
                candidates.append(candidate[1])

        for candidate in self.missing_with_destiny_key:
            if all(not intersect(candidate, current) for current in self.current):
                candidates.append(candidate[0])

        for candidate in self.missing_without_key:
            if all(not intersect(candidate, current) for current in self.current):
                candidates.append(candidate[0])
                candidates.append(candidate[1])
        return candidates

    @staticmethod
    def compute_connections(lhs: list[LightHouse], player_num: int):
        owned_lhs = [lh for lh in lhs if lh.owner == player_num]

        def crosses_lighthouse(candidate_connection):
            for lh in lhs:
                x0, x1 = sorted((candidate_connection[0][0], candidate_connection[1][0]))
                y0, y1 = sorted((candidate_connection[0][1], candidate_connection[1][1]))
                if (x0 <= lh.coor[0] <= x1 and y0 <= lh.coor[1] <= y1
                        and lh.coor not in (candidate_connection[0], candidate_connection[1])
                        and colinear(candidate_connection[0], candidate_connection[1], lh.coor)):
                    return True
            return False

        owned_connections = set()
        missing_connections_origin_key = set()
        missing_connections_destiny_key = set()
        missing_connections_no_key = set()
        for o in range(0, len(owned_lhs) - 1):
            for d in range(o + 1, len(owned_lhs)):
                origin = owned_lhs[o]
                destiny = owned_lhs[d]
                new_connection = (origin.coor, destiny.coor)
                if destiny.coor in origin.connections:
                    owned_connections.add(new_connection)
                elif not crosses_lighthouse(new_connection):
                    if origin.have_key:
                        missing_connections_origin_key.add(new_connection)
                    elif destiny.have_key:
                        missing_connections_destiny_key.add(new_connection)
                    else:
                        missing_connections_no_key.add(new_connection)

        return Connections(owned_connections,
                           missing_connections_origin_key,
                           missing_connections_destiny_key,
                           missing_connections_no_key)
