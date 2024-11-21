#!/usr/bin/python
# -*- coding: utf-8 -*-
# Adapted from https://github.com/marcan/lighthouses_aicontest

from bots import bot
from bots.mocabot_core import Me, Turn, LightHouse


class MocaBot(bot.Bot):
    """Bot that executes random actions"""
    NAME = "MocaBot"

    def play(self, state, step=None):
        actions_list = []
        for i in range(len(state)):
            actions_list.append(self.select_next_action(state[i], self.map[0]))
        return actions_list

    def select_next_action(self, state, game_map):
        me = Me(self.player_num, state["position"], state["energy"], state["view"])
        lhs = [LightHouse(lh["position"], lh["energy"], lh["connections"], lh["owner"], lh["have_key"]) for lh in
               state["lighthouses"]]
        turn = Turn(game_map, me, lhs)
        selected_action, action_data = turn.select_action()

        if selected_action == "attack":
            return self.attack(action_data)
        elif selected_action == "move":
            print(f"BUSCANDO faro desde {me.coor} con {action_data} ")
            return self.move(*action_data)
        elif selected_action == "connect":
            return self.connect(action_data)
        else:
            return self.nop()

