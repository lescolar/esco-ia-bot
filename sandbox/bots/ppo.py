#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from distutils.util import strtobool

from sklearn.preprocessing import StandardScaler
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from bots import bot


# Available actions
ACTIONS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1), "connect", "attack", "pass")
# Choose cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentMLP(nn.Module):
    # Define the neural network for the policy
    def __init__(self, s_size, a_size,):
        super(AgentMLP, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(s_size).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(s_size).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, a_size), std=0.01),
        )
        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(np.array(s_size).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 1), std=1.0),
        # )
        # self.actor = nn.Sequential(
        #     layer_init(nn.Linear(np.array(s_size).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, a_size), std=0.01),
        # )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class AgentCNN(nn.Module):
    def __init__(self, num_maps, a_size: list):
        super(AgentCNN, self).__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=num_maps, out_channels=16, kernel_size=5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32*18*8, 256)), 
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1)
        )

        self.actor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=num_maps, out_channels=16, kernel_size=5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32*18*8, 256)), 
            nn.ReLU(),
            layer_init(nn.Linear(256, a_size), std=0.01)
        )
        # self.critic = nn.Sequential(
        #     layer_init(nn.Conv2d(in_channels=num_maps, out_channels=16, kernel_size=7)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(16, 32, 5)),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     layer_init(nn.Linear(32*33*13, 256)), 
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 1), std=1)
        # )

        # self.actor = nn.Sequential(
        #     layer_init(nn.Conv2d(in_channels=num_maps, out_channels=16, kernel_size=7)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(16, 32, 5)),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     layer_init(nn.Linear(32*13*13, 256)), 
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, a_size), std=0.01)
        # )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PPO(bot.Bot):
    def __init__(self, state_maps, num_envs, num_steps, num_updates, train=True, model_filename='model.pth', use_saved_model=False):
        super().__init__()
        self.NAME = "PPO"
        self.gamma = 0.99 # Discount factor
        self.learning_rate = 2.5e-4 # Learning rate
        self.anneal_lr = True #learning rate annealing for policy and value networks
        self.gae = True # Use GAE for advantage computation
        self.gae_lambda = 0.95 # lambda for the general advantage estimation
        self.num_minibatches = 4 # the number of mini-batches
        self.update_epochs = 4 # the K epochs to update the policy
        self.norm_adv = True # advantages normalization
        self.clip_coef = 0.2 # the surrogate clipping coefficient
        self.clip_vloss = True # whether or not to use a clipped loss for the value function, as per the paper
        self.ent_coef = 0.0 # coefficient of the entropy
        self.vf_coef = 0.5 # coefficient of the value function
        self.max_grad_norm = 0.5 # maximum norm for the gradient clipping
        self.target_kl = None # the target KL divergence threshold
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.update = 0
        self.batch_size = self.num_envs * self.num_steps
        self.use_saved_model = use_saved_model
        self.model_filename = model_filename
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.transitions_temp = []
        self.a_size = len(ACTIONS)
        self.state_maps = state_maps
        self.seed = 1
        self.train = train
        self.save_model = True # Save the model during training
        self.model_path = './artifacts/models'
        self.model_filename = model_filename
        self.use_saved_model = use_saved_model
        self.torch_deterministic = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # random.seed(self.seed)
        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
        # torch.backends.cudnn.deterministic = self.torch_deterministic

        # Initialize tensorboard
        self.writer = SummaryWriter(f"./artifacts/runs")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self).items()])),
        )

    # Initialize agent, optimizer and buffer
    def initialize_game(self, state):
        self.saved_log_probs = []
        if self.state_maps:
            print("Using maps for state: PolicyCNN")
            state = [self.convert_state_cnn(state[i], i) for i in range(len(state))]
            self.num_maps = state[0].shape[2]
            state = np.transpose(state, (0,3,1,2))
            self.s_size = state[0].shape
            self.agent = AgentCNN(self.num_maps, self.a_size).to(self.device)
        else:
            print("Using array for state: PolicyMLP")
            state = [self.convert_state_mlp(state[i]) for i in range(len(state))]
            self.s_size = len(state[0])
            self.agent = AgentMLP(self.s_size, self.a_size).to(self.device)
        if self.train:
            self.initialize_buffer_and_variables()
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate)
        if self.use_saved_model:
            self.load_saved_model()
    
    def initialize_buffer_and_variables(self):
         # Storage setup
        if self.state_maps:
            self.obs = torch.zeros((self.num_steps, self.num_envs) + self.s_size).to(device)
        else:    
            self.obs = torch.zeros((self.num_steps, self.num_envs) + (self.s_size,)).to(device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + ()).to(device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.global_step = 0
        self.start_time = time.time()

    def initialize_experience_gathering(self):
        # Annealing the rate if instructed to do so.
        if self.anneal_lr:
            frac = 1.0 - (self.update - 1.0) / self.num_updates
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow
    
    def convert_state_mlp(self, state):
        # Create array for view data
        view = []
        for i in range(len(state['view'])):
            view = view + state['view'][i]
        cx = state['position'][0]
        cy = state['position'][1]
        cx_min, cx_max = cx-3, cx+3
        cy_min, cy_max = cy-3, cy+3
        lighthouses = np.zeros((7,7), dtype=int)
        lighthouses_dict = dict((tuple(lh["position"]), lh['energy']) for lh in state["lighthouses"])
        for key in lighthouses_dict.keys():
            if cx_min <= key[0] <= cx_max and cy_min <= key[1] <= cy_max:
                lighthouses[key[0]+3-cx, key[1]+3-cy] = lighthouses_dict[key] + 1
        lighthouses_info = []
        # Create array for lighthouses data (within 3 steps of the bot)
        for i in range(len(lighthouses)):
            lighthouses_info = lighthouses_info + list(lighthouses[i])
        new_state = np.array([state['position'][0], state['position'][1], state['energy'], len(state['lighthouses'])] + view + lighthouses_info)
        sc = StandardScaler()
        new_state = sc.fit_transform(new_state.reshape(-1, 1))
        new_state = new_state.squeeze()
        return new_state
    
    def z_score_scaling(self, arr):
        arr_mean = np.mean(arr)
        arr_std = np.std(arr)
        scaled_arr = (arr - arr_mean) / arr_std
        return scaled_arr

    def convert_state_cnn(self, state, i):
        # Create base layer that will serve as the base for all layers of the state
        # This layer has zeros in all cells except the border cells in which the value is -1
        base_layer = np.array(self.map[i].copy())
        base_layer = np.transpose(base_layer)

        # Create player layer which has the value of the energy of the player + 1 where the player is located
        # 1 is added to the energy to cover the case that the energy of the player is 0
        player_layer = base_layer.copy()
        x, y = state['position'][0], state['position'][1]
        player_layer[x,y] = 1 + state['energy']
        player_layer = self.z_score_scaling(player_layer)
    
        # Create view layer with energy level near the player
        view_layer = base_layer.copy()
        state['view'] = np.array(state['view'])
        start_row, start_col = x-3, y-3
        if y+3 > view_layer.shape[1]-1:
            adjust = view_layer.shape[1]-1 - (y+3)
            state['view'] = state['view'][:,:adjust]
        if x+3 > view_layer.shape[0]-1:
            adjust = view_layer.shape[0]-1 - (x+3)
            state['view'] = state['view'][:adjust,:]
        if y-3 < 0:
            adjust = 3-y
            state['view'] = state['view'][:,adjust:]
            start_col = 0
        if x-3 < 0:
            adjust = 3-x
            state['view'] = state['view'][adjust:,:]
            start_row = 0
        view_layer[start_row:start_row+state['view'].shape[0], start_col:start_col+state['view'].shape[1]] = state['view']
        view_layer = self.z_score_scaling(view_layer)

        # Create layer that has the energy of the lighthouse + 1 where the lighthouse is located
        # 1 is added to the lighthouse energy to cover the case that the energy of the lighthouse is 0
        lh_energy_layer = base_layer.copy()
        lh = state['lighthouses']
        for i in range(len(lh)):
            x, y = lh[i]['position'][0], lh[i]['position'][1]
            lh_energy_layer[x,y] = 1 + lh[i]['energy']
        lh_energy_layer = self.z_score_scaling(lh_energy_layer)

        # Create layer that has the number of the player that controls each lighthouse
        # If no player controls the lighthouse, then a value of -1 is assigned
        lh_control_layer = base_layer.copy()
        for i in range(len(lh)):
            x, y = lh[i]['position'][0], lh[i]['position'][1]
            if lh[i]['owner'] == None:
                lh_control_layer[x,y] = -1
            else:
                lh_control_layer[x,y] = lh[i]['owner']
        lh_control_layer = self.z_score_scaling(lh_control_layer)

        # Create layer that indicates the lighthouses that are connected
        # If the lighthouse is not connected, then a value of -1 is assigned, if it is connected then it is 
        # assigned the number of connections that it has
        lh_connections_layer = base_layer.copy()
        for i in range(len(lh)):
            x, y = lh[i]['position'][0], lh[i]['position'][1]
            if len(lh[i]['connections']) == 0:
                lh_connections_layer[x,y] = -1
            else:
                lh_connections_layer[x,y] = len(lh[i]['connections'])
        lh_connections_layer = self.z_score_scaling(lh_connections_layer)

        # Create layer that indicates if the player has the key to the light house
        # Assign value of 1 if has key and -1 if does not have key
        lh_key_layer = base_layer.copy()
        for i in range(len(lh)):
            x, y = lh[i]['position'][0], lh[i]['position'][1]
            if lh[i]['have_key']:
                lh_key_layer[x,y] = 1
            else:
                lh_key_layer[x,y] = -1
        lh_key_layer = self.z_score_scaling(lh_key_layer)

        # Concatenate the maps into one state
        player_layer = np.expand_dims(player_layer, axis=2)
        view_layer = np.expand_dims(view_layer, axis=2)
        lh_energy_layer = np.expand_dims(lh_energy_layer, axis=2)
        lh_control_layer = np.expand_dims(lh_control_layer, axis=2)
        lh_connections_layer = np.expand_dims(lh_connections_layer, axis=2)
        lh_key_layer = np.expand_dims(lh_key_layer, axis=2)

        new_state = np.concatenate((player_layer, view_layer, lh_energy_layer, lh_connections_layer, lh_control_layer, lh_key_layer), axis=2) 
        return new_state
    
    def valid_lighthouse_connections(self, state):
        # Check if exist possible lighthouse connections
        cx = state['position'][0]
        cy = state['position'][1]
        lighthouses = dict((tuple(lh["position"]), lh) for lh in state["lighthouses"])
        possible_connections = []
        if (cx, cy) in lighthouses:
            if lighthouses[(cx, cy)]["owner"] == self.player_num:
                for dest in lighthouses.keys():
                    if (dest != (cx, cy) and lighthouses[dest]["have_key"] and
                        [cx, cy] not in lighthouses[dest]["connections"] and
                        lighthouses[dest]["owner"] == self.player_num):
                        possible_connections.append(dest)
        return possible_connections

    def play(self, state, step=None):
        if self.train:
            self.global_step += 1 * self.num_envs
        actions_list = []
        if self.state_maps:
            new_state = [self.convert_state_cnn(state[i], i) for i in range(len(state))]
            new_state = np.transpose(new_state, (0,3,1,2))
        else:
            new_state = [self.convert_state_mlp(state[i]) for i in range(len(state))]
        new_state = torch.from_numpy(np.array(new_state)).float().to(device) 
        if self.train:
            self.obs[step] = new_state
        with torch.no_grad():
            action, log_prob, _, value = self.agent.get_action_and_value(new_state)
            if self.train:
                self.saved_log_probs.append(log_prob)
                self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = log_prob
        for i in range(len(action)):
            # Move
            if ACTIONS[action[i]] != "attack" and ACTIONS[action[i]] != "connect" and ACTIONS[action[i]] != "pass":
                actions_list.append(self.move(*ACTIONS[action[i]]))
            # Pass
            elif ACTIONS[action[i]] == "pass":
                actions_list.append(self.nop())
            # Attack
            elif ACTIONS[action[i]] == "attack":
                actions_list.append(self.attack(state[i]['energy']))
            # Connect
            elif ACTIONS[action[i]] == "connect":
                # TODO: improve selection of lighthouse connection, right now uses function to select them
                possible_connections = self.valid_lighthouse_connections(state[i])
                if not possible_connections:
                    actions_list.append(self.nop()) #pass the turn
                else: 
                    actions_list.append(self.connect(random.choice(possible_connections)))
        return actions_list

    def calculate_advantage(self, next_obs):
            # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            if self.gae:
                self.advantages = torch.zeros_like(self.rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    self.advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                self.returns = self.advantages + self.values
            else:
                self.returns = torch.zeros_like(self.rewards).to(device)
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_return = self.returns[t + 1]
                    self.returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
                self.advantages = self.returns - self.values
            print("advantages: ", self.advantages.sum(axis=0))

    def optimize_model(self, transitions):
        # flatten the batch
        next_obs = transitions[-1][3]
        if self.state_maps:
            next_obs = [self.convert_state_cnn(next_obs[i], i) for i in range(len(next_obs))]
            next_obs = np.transpose(next_obs, (0,3,1,2))
        else:
            next_obs = [self.convert_state_mlp(next_obs[i]) for i in range(len(next_obs))]
        next_obs = torch.from_numpy(np.array(next_obs)).float().to(device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        for i in range(len(transitions)):
            self.rewards[i] = torch.tensor(transitions[i][2]).to(device).view(-1)
        print("rewards: ", self.rewards.sum(axis=0))
        self.calculate_advantage(next_obs)
        if self.state_maps:
            b_obs = self.obs.reshape((-1,) + self.s_size)
        else:
            b_obs = self.obs.reshape((-1,) + (self.s_size,))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + ())
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # Record losses, learning rate
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
        self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
        print("policy loss: ", pg_loss.item())
        print("value loss: ", v_loss.item())

    def save_trained_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self.agent.state_dict(), os.path.join(self.model_path, self.model_filename))

    def load_saved_model(self):
        if self.model_filename and os.path.isfile(os.path.join(self.model_path, self.model_filename)):
            self.agent.load_state_dict(torch.load(os.path.join(self.model_path, self.model_filename)))
            print(f"Loaded saved model: {self.model_filename}")
        else:
            print("No saved model")
