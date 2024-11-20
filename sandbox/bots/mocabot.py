#!/usr/bin/python
# -*- coding: utf-8 -*-
# Adapted from https://github.com/marcan/lighthouses_aicontest

import random

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

        lhs = state["lighthouses"]
        lh_paths = dict(
            (tuple(lh["position"]), pathfinder.find_path(game_map, energy_per_turn_grid, (cx, cy), lh['position']))
            for lh in lhs)

        owned_lhs = [lh for lh in lhs if lh['owner'] == self.player_num]
        current_connections = set()
        missing_connections_origin_key = set()
        missing_connections_destiny_key = set()
        missing_connections_no_key = set()

        def can_connect(c):
            return all(not intersect(c, current) for current in current_connections)

        for o in range(0, len(owned_lhs) - 1):
            for d in range(o + 1, len(owned_lhs)):
                origin = owned_lhs[o]
                destiny = owned_lhs[d]
                if destiny["position"] in origin["connections"]:
                    current_connections.add((origin["position"], destiny["position"]))
                elif origin["have_key"]:
                    missing_connections_origin_key.add((origin["position"], destiny["position"]))
                elif destiny["have_key"]:
                    missing_connections_destiny_key.add((origin["position"], destiny["position"]))
                else:
                    missing_connections_no_key.add((origin["position"], destiny["position"]))

        for y in range(len(energy_per_turn_grid)):
            for x in range(len(energy_per_turn_grid[0])):
                if abs(y - cy) <= 3 and abs(x - cx) <= 3:
                    current_energy_grid[y][x] = state["view"][y - cy + 3][x - cx + 3]

        # If there is a lighthouse in the current position
        if (cx, cy) in lighthouses.keys():
            # Probability 60%: connect to valid remote lighthouse
            if lighthouses[(cx, cy)]["owner"] == self.player_num:
                for dest in lighthouses.keys():
                    # Do not connect to itself
                    # Do not connect if there is no key available
                    # Do not connect if there is an existing connection
                    # Do not connect if the destination is not controlled
                    if (dest != (cx, cy) and
                            lighthouses[dest]["have_key"] and
                            (cx, cy) not in lighthouses[dest]["connections"] and
                            lighthouses[dest]["owner"] == self.player_num and
                            can_connect(((cx, cy), dest))):
                        print(f"CONECTANDO desde {(cx, cy)} con el faro {dest}")
                        return self.connect(dest)
            elif state["energy"] > lighthouses[(cx, cy)]["energy"]:
                print(f"ATACANDO faro {(cx, cy)} con energía {lighthouses[(cx, cy)]["energy"]}")
                return self.attack(state["energy"])

        # Make triangles
        if len(owned_lhs) >= 3:

            for candidate in missing_connections_origin_key:
                if all(not intersect(candidate, current) for current in current_connections):
                    (destiny_path, _) = lh_paths[candidate[1]]
                    move = (destiny_path[1][0] - destiny_path[0][0], destiny_path[1][1] - destiny_path[0][1])
                    print(f"YENDO desde {(cx, cy)} a faro {candidate[1]} para cerrar")
                    return self.move(*move)

            for candidate in missing_connections_destiny_key:
                if all(not intersect(candidate, current) for current in current_connections):
                    (origin_path, _) = lh_paths[candidate[0]]
                    move = (origin_path[1][0] - origin_path[0][0], origin_path[1][1] - origin_path[0][1])
                    print(f"YENDO desde {(cx, cy)} a faro {candidate[0]} para cerrar")
                    return self.move(*move)

            for candidate in missing_connections_no_key:
                if all(not intersect(candidate, current) for current in current_connections):
                    if candidate[0] == (cx, cy):
                        print(f"Esperando en {(cx, cy)} la llave")
                        return self.nop()
                    (origin_path, _) = lh_paths[candidate[0]]
                    move = (origin_path[1][0] - origin_path[0][0], origin_path[1][1] - origin_path[0][1])
                    print(f"YENDO desde {(cx, cy)} a faro {candidate[0]} para cerrar")
                    return self.move(*move)

            # for o in range(0, len(owned_lhs) - 1):
            #     for d in range(o + 1, len(owned_lhs)):
            #         origin = owned_lhs[o]
            #         destiny = owned_lhs[d]
            #         if origin["position"] not in destiny["connections"]:
            #             if origin["have_key"]:
            #                 if (cx, cy) == destiny['position']:
            #                     print(f"CONECTANDO desde {(cx, cy)} con el faro {origin}")
            #                     return self.connect(origin)
            #                 else:
            #                     (path, _) = pathfinder.find_path(game_map, energy_per_turn_grid, (cx, cy),
            #                                                      destiny['position'])
            #                     move = (path[1][0] - path[0][0], path[1][1] - path[0][1])
            #                     print(f"YENDO desde {(cx, cy)} a faro {destiny} para cerrar")
            #                     return self.move(*move)
            #             elif destiny["have_key"]:
            #                 if (cx, cy) == origin['position']:
            #                     print(f"CONECTANDO desde {(cx, cy)} con el faro {destiny}")
            #                     return self.connect(destiny)
            #                 else:
            #                     (path, _) = pathfinder.find_path(game_map, energy_per_turn_grid, (cx, cy),
            #                                                      origin['position'])
            #                     move = (path[1][0] - path[0][0], path[1][1] - path[0][1])
            #                     print(f"YENDO desde {(cx, cy)} a faro {destiny} para origin")
            #                     return self.move(*move)
            #             elif origin['position'] == (cx, cy):
            #                 (path_destiny, _) = pathfinder.find_path(game_map, energy_per_turn_grid, (cx, cy),
            #                                                          destiny['position'])
            #                 move = (path_destiny[1][0] - path_destiny[0][0], path_destiny[1][1] - path_destiny[0][1])
            #                 print(f"YENDO desde {(cx, cy)} a faro {destiny} para destiny")
            #                 return self.move(*move)
            #             elif destiny['position'] == (cx, cy):
            #                 (path_origin, _) = pathfinder.find_path(game_map, energy_per_turn_grid, (cx, cy),
            #                                                         origin['position'])
            #                 move = (path_origin[1][0] - path_origin[0][0], path_origin[1][1] - path_origin[0][1])
            #                 print(f"YENDO desde {(cx, cy)} a faro {origin} para origin")
            #                 return self.move(*move)
            #             else:
            #                 (path_origin, _) = pathfinder.find_path(game_map, energy_per_turn_grid, (cx, cy),
            #                                                         origin['position'])
            #                 (path_destiny, _) = pathfinder.find_path(game_map, energy_per_turn_grid, (cx, cy),
            #                                                          destiny['position'])
            #                 if len(path_origin) < len(path_destiny):
            #                     move = (path_origin[1][0] - path_origin[0][0], path_origin[1][1] - path_origin[0][1])
            #                     print(f"YENDO desde {(cx, cy)} a faro {origin} para origin")
            #                     return self.move(*move)
            #                 else:
            #                     move = (
            #                     path_destiny[1][0] - path_destiny[0][0], path_destiny[1][1] - path_destiny[0][1])
            #                     print(f"YENDO desde {(cx, cy)} a faro {destiny} para destiny")
            #                     return self.move(*move)

        # Move to an unowned
        unowned_lhs = [lh for lh in lhs if (lh['owner'] != self.player_num and state["energy"] >= lh["energy"])]
        paths = [pathfinder.find_path(game_map, energy_per_turn_grid, (cx, cy), lh['position']) for lh in unowned_lhs]
        energetic_paths = [p for (p, r) in paths if len(p) * 10 < (state["energy"] + r)]
        if len(energetic_paths) > 0:
            shortest_path = sorted(energetic_paths, key=lambda p: len(p))[0]
        else:
            shortest_path = sorted([p for (p, r) in paths], key=lambda p: len(p))[0]
        # Check if the move is valid
        move = (shortest_path[1][0] - shortest_path[0][0], shortest_path[1][1] - shortest_path[0][1])
        print(f"BUSCANDO faro desde {(cx, cy)} con {move} ")
        return self.move(*move)
