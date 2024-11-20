#!/usr/bin/python
# -*- coding: utf-8 -*-
# Adapted from https://github.com/marcan/lighthouses_aicontest


class Bot(object):
    """Base bot. It does nothing (passing all turns)"""
    NAME = "NullBot"

    def __init__(self):
        self.transitions = []
        self.transitions_temp = []
        self.scores = []
        self.player_num = 0
        self.game_map = None
        self.save_model = []
        self.final_scores_list = []
        self.last_episode_score = 0
        self.policy_loss_list = []
        self.update = 0

    def initialize_game(self, state):
        # print(self.__dict__)
        print(state)
        pass

    def initialize_experience_gathering(self):
        pass

    def play(self, state):
        """Play: it is executed each turn.
        It must return an action.
        state: current state of the game.
        """
        return self.nop()
    
    def optimize_model(self, transitions):
        pass

    def save_trained_model(self):
        pass

    def success(self):
        """Executed when the previous action is valid"""
        pass

    def error(self, message, last_move):
        """Executed when the previous action is not valid"""
        print("Recibido error: %s", message)
        print("Jugada previa: %r", last_move)

    # ==========================================================================
    # Possible moves. Do not overwrite or modify.
    # ==========================================================================

    def nop(self):
        """Pass turn"""
        return {
            "command": "pass",
        }

    def move(self, x, y):
        """Move to a specific position
        x: delta x (0, -1, 1)
        y: delta y (0, -1, 1)
        """
        return {
            "command": "move",
            "x": x,
            "y": y
        }

    def attack(self, energy):
        """Attack a lighthouse
        energy: energy used on the attack (positive integer)
        """
        return {
            "command": "attack",
            "energy": energy
        }

    def connect(self, destination):
        """Connect remote lighthouse
        destination: tuple o list (x,y): coordinates of the remote lighthouse
        """
        return {
            "command": "connect",
            "destination": destination
        }