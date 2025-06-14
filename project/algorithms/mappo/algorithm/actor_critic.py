import torch
import torch.nn as nn
from project.algorithms.utils.util import init, check
from project.utils.util import get_shape_from_obs_space
from project.algorithms.utils.mlp import MLPBase
from project.algorithms.utils.rnn import RNNLayer
from project.algorithms.utils.act import ACTLayer
from project.algorithms.utils.popart import PopArt
from project.algorithms.utils.gat import GAT
from project.algorithms.utils.link import Link


class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N,
                            self._use_orthogonal)
        self.act = ACTLayer(action_space, self.hidden_size,
                            self._use_orthogonal, self._gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(
            actor_features, rnn_states, masks)
        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(
            actor_features, rnn_states, masks)
        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks
            else None)
        return action_log_probs, dist_entropy


class Critic(nn.Module):
    def __init__(self, args, cent_obs_space_, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self._use_GTDE = args.use_GTDE
        self._GAT_dim = args.gat_dim
        self._attention_head = args.attention_head
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][self._use_orthogonal]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space_)
        self.ally_features = cent_obs_space_[1]
        self.base = MLPBase(args, cent_obs_shape)
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N,
                            self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_GTDE:
            self.GAT = GAT(self.hidden_size, self._GAT_dim,
                           self.hidden_size, self._attention_head)
            self.link = Link(args, cent_obs_space_, self.hidden_size)
            if self._use_popart:
                self.v_out = init_(
                    PopArt(self.hidden_size + self.ally_features[0] + 1, 1, device=device))
            else:
                self.v_out = init_(
                    nn.Linear(self.hidden_size + self.ally_features[0] + 1, 1))
        else:
            if self._use_popart:
                self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        critic_features = self.base(cent_obs)
        critic_features, rnn_states = self.rnn(
            critic_features, rnn_states, masks)
        if self._use_GTDE:
            edge_matrix = self.link(critic_features.detach())

            critic_features = critic_features.reshape(
                -1, (self.ally_features[0] + 1), critic_features.shape[-1])
            critic_features = self.GAT(critic_features,
                                       edge_matrix.reshape(-1, (self.ally_features[0] + 1),
                                                           (self.ally_features[
                                                               0] + 1)).detach())
            critic_features = torch.cat(
                [critic_features.reshape(-1, critic_features.shape[-1]), edge_matrix], dim=-1)
        value = self.v_out(critic_features)
        return value, rnn_states
