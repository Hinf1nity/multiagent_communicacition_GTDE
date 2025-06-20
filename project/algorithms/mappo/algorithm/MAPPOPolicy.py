import torch
from project.algorithms.mappo.algorithm.actor_critic import Actor, Critic
from project.utils.util import update_linear_schedule


class MAPPOPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        self.update_learning_rate(
            self.actor_optimizer, episode, episodes, self.lr)
        self.update_learning_rate(
            self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,
                    masks, available_actions=None,
                    deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic)
        values, rnn_states_critic = self.critic(
            cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_state_critic, action, masks,
                         available_actions=None,
                         active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks)
        values, _ = self.critic(cent_obs, rnn_state_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks,
            available_actions=None,
            deterministic=False):
        actions, _, rnn_states_actor = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic)
        return actions, rnn_states_actor
