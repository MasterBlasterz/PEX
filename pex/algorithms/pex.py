import copy
import torch

from pex.utils.util import (DEFAULT_DEVICE, epsilon_greedy_sample,
                            extract_sub_dict)
from pex.algorithms.iql import IQL, EXP_ADV_MAX


class PEX(IQL):
    def __init__(self, critic, vf, policy, optimizer_ctor,
                 tau, beta, discount, target_update_rate, ckpt_path, inv_temperature,
                 copy_to_target=False):
        super().__init__(critic=critic, vf=vf, policy=policy,
                         optimizer_ctor=optimizer_ctor,
                         max_steps=None,
                         tau=tau, beta=beta,
                         discount=discount,
                         target_update_rate=target_update_rate,
                         use_lr_scheduler=False)

        self.policy_offline = copy.deepcopy(self.policy).to(DEFAULT_DEVICE)

        self._inv_temperature = inv_temperature

        # load checkpoint if ckpt_path is not None
        if ckpt_path is not None:

            map_location = None
            if not torch.cuda.is_available():
                map_location = torch.device('cpu')
            checkpoint = torch.load(ckpt_path, map_location=map_location)

            # extract sub-dictionary
            policy_state_dict = extract_sub_dict("policy", checkpoint)
            critic_state_dict = extract_sub_dict("critic", checkpoint)

            self.policy_offline.load_state_dict(policy_state_dict)
            self.critic.load_state_dict(critic_state_dict)
            self.vf.load_state_dict(extract_sub_dict("vf", checkpoint))

            if copy_to_target:
                self.target_critic.load_state_dict(critic_state_dict)
            else:
                target_critic_state_dict = extract_sub_dict("target_critic", checkpoint)
                self.target_critic.load_state_dict(target_critic_state_dict)

        
    def select_action(self, observations, evaluate=False, return_all_actions=False,
                       return_policy_selection=False, adaptive_comp='greedy'):
        """
        adaptive_comp:
            'greedy' - value-weighted softmax composition (original PEX), sampled
                       with epsilon-greedy exploration on top.
            'uni'    - uniform distribution over {offline, online} policy, ignoring Q.
            'hard'   - deterministic hard-max over Q(s, a_i); no stochastic mixing.
        """
        is_batch = (observations.dim() == 2)

        observations = observations.unsqueeze(0)
        a1 = self.policy_offline.act(observations, deterministic=True)

        dist = self.policy(observations)
        if evaluate:
            a2 = epsilon_greedy_sample(dist, eps=0.1)
        else:
            a2 = epsilon_greedy_sample(dist, eps=1.0)

        q1 = self.critic.min(observations, a1)
        q2 = self.critic.min(observations, a2)

        q = torch.stack([q1, q2], dim=-1)

        if adaptive_comp == 'greedy':
            logits = q * self._inv_temperature
            w_dist = torch.distributions.Categorical(logits=logits)
            if evaluate:
                w = epsilon_greedy_sample(w_dist, eps=0.1)
            else:
                w = epsilon_greedy_sample(w_dist, eps=1.0)

        elif adaptive_comp == 'uni':
            # Uniform 50/50 selection, ignoring Q entirely.
            probs = torch.full_like(q, 0.5)
            w_dist = torch.distributions.Categorical(probs=probs)
            w = w_dist.sample()

        elif adaptive_comp == 'hard':
            # Deterministic argmax over Q, no stochastic mixing.
            w = torch.argmax(q, dim=-1)

        else:
            raise ValueError(f"Unknown adaptive_comp mode: {adaptive_comp}")

        # jk59: get selected policy (0 for offline, 1 for online)
        if return_policy_selection:
            policy_choice = w.cpu().tolist() if is_batch else int(w.item())

        w = w.unsqueeze(-1)

        action = (1 - w) * a1 + w * a2

        if not return_all_actions:
            if return_policy_selection:
                return action.squeeze(0), policy_choice
            else:
                return action.squeeze(0)

        elif return_policy_selection:
            return action.squeeze(0), a1.squeeze(0), a2.squeeze(0), policy_choice
        else:
            return action.squeeze(0), a1.squeeze(0), a2.squeeze(0)


    def policy_update(self, observations, adv, actions):
        actions = self.select_action(observations)
        with torch.no_grad():
            target_q = self.target_critic.min(observations, actions)
        v = self.vf(observations)
        adv = target_q.detach() - v
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        bc_losses = -policy_out.log_prob(actions.detach())

        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.use_lr_scheduler:
            self.policy_lr_schedule.step()
