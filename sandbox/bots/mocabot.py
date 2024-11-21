#!/usr/bin/python
# -*- coding: utf-8 -*-
# Adapted from https://github.com/marcan/lighthouses_aicontest

from bots import bot
from bots import pathfinder
from bots.geom import intersect
from bots.map_utils import energy_per_turn


class MocaBot(bot.Bot):
    """Bot that executes random actions"""
    NAME = "MocaBot"

    def play(self, state, step=None):
        actions_list = []
        for i in range(len(state)):
            actions_list.append(self.select_next_action(state[i], self.map[0]))
        return actions_list

    def select_next_action(self, state, game_map):
        cx, cy = state["position"]
        lighthouses = dict((tuple(lh["position"]), lh)
                           for lh in state["lighthouses"])
        energy_per_turn_grid = energy_per_turn(game_map, lighthouses.keys())
        current_energy_grid = [[-1] * len(energy_per_turn_grid[0]) for _ in range(len(energy_per_turn_grid))]

        for y in range(len(energy_per_turn_grid)):
            for x in range(len(energy_per_turn_grid[0])):
                if abs(y - cy) <= 3 and abs(x - cx) <= 3:
                    current_energy_grid[y][x] = state["view"][y - cy + 3][x - cx + 3]

        lhs = state["lighthouses"]
        lh_paths = LightHousePaths.compute_connections(game_map,
                                                       energy_per_turn_grid,
                                                       current_energy_grid,
                                                       (cx, cy),
                                                       state["lighthouses"])

        owned_lhs = [lh for lh in lhs if lh['owner'] == self.player_num]
        connections = Connections.compute_connections(lh_paths, owned_lhs)

        # If there is a lighthouse in the current position
        if (cx, cy) in lighthouses.keys():
            # Probability 60%: connect to valid remote lighthouse
            if lighthouses[(cx, cy)]["owner"] == self.player_num:
                for dest in lighthouses.keys():
                    if (
                            dest != (cx, cy)
                            and lighthouses[dest]["have_key"]
                            and (cx, cy) not in lighthouses[dest]["connections"]
                            and lighthouses[dest]["owner"] == self.player_num
                            and connections.can_connect(((cx, cy), dest))
                    ):
                        print(f"CONECTANDO desde {(cx, cy)} con el faro {dest}")
                        return self.connect(dest)
            elif state["energy"] > lighthouses[(cx, cy)]["energy"]:
                print(f"ATACANDO faro {(cx, cy)} con energía {lighthouses[(cx, cy)]["energy"]}")
                return self.attack(state["energy"])

        # Make triangles
        if len(owned_lhs) >= 3:
            candidate = connections.search_for_closable_connection()
            if candidate:
                if candidate == (cx, cy):
                    print(f"Esperando en {(cx, cy)} la llave")
                    return self.nop()
                else:
                    move = lh_paths.move_towards(candidate)
                    print(f"YENDO desde {(cx, cy)} a faro {candidate} para cerrar")
                    return self.move(*move)

        # Move to an unowned
        unowned_lhs = [lh for lh in lhs if (lh['owner'] != self.player_num and state["energy"] >= lh["energy"])]
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
                return self.nop()
            else:
                print(f"BUSCANDO faro desde {(cx, cy)} con {best_direction} ")
                return self.move(*best_direction)
        else:
            paths = [lh_paths.paths_dict[lh['position']] for lh in unowned_lhs]
            energetic_paths = [p for (p, r) in paths if len(p) * 10 < (state["energy"] + r)]
            if len(energetic_paths) > 0:
                shortest_path = sorted(energetic_paths, key=lambda p: len(p))[0]
            else:
                shortest_path = sorted([p for (p, r) in paths], key=lambda p: len(p))[0]
            # Check if the move is valid
            move = (shortest_path[1][0] - shortest_path[0][0], shortest_path[1][1] - shortest_path[0][1])
            print(f"BUSCANDO faro desde {(cx, cy)} con {move} ")
            return self.move(*move)


class LightHousePaths:
    def __init__(self, paths_dict):
        self.paths_dict = paths_dict

    def get_path_len(self, destiny):
        return len(self.paths_dict[destiny][0])

    def move_towards(self, destiny):
        (computed_path, _) = self.paths_dict[destiny]
        return computed_path[1][0] - computed_path[0][0], computed_path[1][1] - computed_path[0][1]

    @staticmethod
    def compute_connections(grid, energy_per_turn_grid, current_energy_grid, origin, lighthouses):
        lh_paths = dict(
            (tuple(lh["position"]),
             pathfinder.find_path(grid, energy_per_turn_grid, current_energy_grid, origin, lh['position']))
            for lh in lighthouses)
        return LightHousePaths(lh_paths)


class Connections:
    def __init__(self, current, missing_with_origin_key, missing_with_destiny_key, missing_without_key):
        self.current = current
        self.missing_with_origin_key = missing_with_origin_key
        self.missing_with_destiny_key = missing_with_destiny_key
        self.missing_without_key = missing_without_key

    def can_connect(self, candidate):
        return all(not intersect(candidate, current) for current in self.current)

    def search_for_closable_connection(self):
        for candidate in self.missing_with_origin_key:
            if all(not intersect(candidate, current) for current in self.current):
                return candidate[1]

        for candidate in self.missing_with_destiny_key:
            if all(not intersect(candidate, current) for current in self.current):
                return candidate[0]

        for candidate in self.missing_without_key:
            if all(not intersect(candidate, current) for current in self.current):
                return candidate[0]
        return None

    @staticmethod
    def compute_connections(lh_paths, owned_lhs):
        current_connections = set()
        missing_connections_origin_key = set()
        missing_connections_destiny_key = set()
        missing_connections_no_key = set()
        for o in range(0, len(owned_lhs) - 1):
            for d in range(o + 1, len(owned_lhs)):
                origin = owned_lhs[o]
                destiny = owned_lhs[d]
                new_connection = (origin["position"], destiny["position"])
                if destiny["position"] in origin["connections"]:
                    current_connections.add(new_connection)
                elif origin["have_key"]:
                    missing_connections_origin_key.add(new_connection)
                elif destiny["have_key"]:
                    missing_connections_destiny_key.add(new_connection)
                else:
                    missing_connections_no_key.add(new_connection)

        missing_connections_origin_key = sorted(missing_connections_origin_key,
                                                key=lambda con: lh_paths.get_path_len(con[1]))
        missing_connections_destiny_key = sorted(missing_connections_destiny_key,
                                                 key=lambda con: lh_paths.get_path_len(con[0]))
        missing_connections_no_key = sorted(missing_connections_no_key,
                                            key=lambda con: min(lh_paths.get_path_len(con[0]),
                                                                lh_paths.get_path_len(con[1])))
        return Connections(current_connections,
                           missing_connections_origin_key,
                           missing_connections_destiny_key,
                           missing_connections_no_key)
