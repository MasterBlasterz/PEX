import copy
import torch

from pex.utils.util import (DEFAULT_DEVICE, epsilon_greedy_sample,
                            extract_sub_dict)
from pex.algorithms.iql import IQL, EXP_ADV_MAX


# Example joint-group partition.
HUMANOID_JOINT_GROUPS = {
    "abdomen": [0, 1, 2],                           # torso
    "hip": [3, 4, 5, 7, 8, 9],                      # hips 
    "knee": [6, 10],                                # knees
    "shoulder": [11, 12, 14, 15],                   # shoulders
    "elbow": [13, 16],                              # elbows
}


class PEXGrouped(IQL):
    """
    Ablation of PEX: instead of selecting one whole action vector from
    {offline policy, online policy} per state (original PEX), select
    independently per joint-group, using marginal Q-swaps evaluated
    against the existing (non-factored) critic.

    Everything else (offline pretraining, critic transfer, buffer sharing,
    freezing pi_beta) is identical to PEX — only composition granularity
    changes, so this isolates that one factor cleanly.
    """

    def __init__(self, critic, vf, policy, optimizer_ctor,
                 tau, beta, discount, target_update_rate, ckpt_path, inv_temperature,
                 joint_groups=None, copy_to_target=False):
        super().__init__(critic=critic, vf=vf, policy=policy,
                         optimizer_ctor=optimizer_ctor,
                         max_steps=None,
                         tau=tau, beta=beta,
                         discount=discount,
                         target_update_rate=target_update_rate,
                         use_lr_scheduler=False)

        self.policy_offline = copy.deepcopy(self.policy).to(DEFAULT_DEVICE)
        self._inv_temperature = inv_temperature

        self.joint_groups = joint_groups or HUMANOID_JOINT_GROUPS
        self._group_list = list(self.joint_groups.values())

        if ckpt_path is not None:
            map_location = None if torch.cuda.is_available() else torch.device('cpu')
            checkpoint = torch.load(ckpt_path, map_location=map_location)

            policy_state_dict = extract_sub_dict("policy", checkpoint)
            critic_state_dict = extract_sub_dict("critic", checkpoint)

            self.policy_offline.load_state_dict(policy_state_dict)
            self.critic.load_state_dict(critic_state_dict)
            self.vf.load_state_dict(extract_sub_dict("vf", checkpoint))

            if copy_to_target:
                self.target_critic.load_state_dict(critic_state_dict)
            else:
                self.target_critic.load_state_dict(extract_sub_dict("target_critic", checkpoint))

    def select_action(self, observations, evaluate=False,
                       return_all_actions=False, return_policy_selection=False, ablation='none'):
        is_batch = (observations.dim() == 2)
        observations = observations.unsqueeze(0)

        a1 = self.policy_offline.act(observations, deterministic=True)  # [B,1,D] offline
        dist = self.policy(observations)
        eps = 0.1 if evaluate else 1.0
        a2 = epsilon_greedy_sample(dist, eps=eps)                       # [B,1,D] online

        # --- Per-group marginal swap evaluation ---
        # Start from a1 as the base action, then test swapping each group
        # to a2's values, holding everything else at a1.
        base_action = a1.clone()
        group_choice = torch.zeros(
            a1.shape[:-1] + (len(self._group_list),),
            device=a1.device, dtype=torch.long
        )  # [B,1,num_groups] which policy each group came from (0=offline,1=online)

        final_action = a1.clone()

        for g_idx, dims in enumerate(self._group_list):
            dims_t = torch.tensor(dims, device=a1.device)

            # Candidate: base action with this group replaced by a2's values
            candidate = base_action.clone()
            candidate[..., dims_t] = a2[..., dims_t]

            q_base = self.critic.min(observations, base_action)   # group = offline
            q_cand = self.critic.min(observations, candidate)     # group = online

            q_group = torch.stack([q_base, q_cand], dim=-1)
            logits = q_group * self._inv_temperature
            w_dist = torch.distributions.Categorical(logits=logits)
            w_eps = 0.1 if evaluate else 1.0
            w_g = epsilon_greedy_sample(w_dist, eps=w_eps)

            group_choice[..., g_idx] = w_g

            w_g_expanded = w_g.unsqueeze(-1).float()  # [B,1,1]
            final_action[..., dims_t] = (
                (1 - w_g_expanded) * a1[..., dims_t]
                + w_g_expanded * a2[..., dims_t]
            )

        if return_policy_selection:
            policy_choice = (
                group_choice.cpu().tolist() if is_batch
                else group_choice.squeeze(0).squeeze(0).tolist()
            )

        if not return_all_actions:
            if return_policy_selection:
                return final_action.squeeze(0), policy_choice
            else:
                return final_action.squeeze(0)
        elif return_policy_selection:
            return final_action.squeeze(0), a1.squeeze(0), a2.squeeze(0), policy_choice
        else:
            return final_action.squeeze(0), a1.squeeze(0), a2.squeeze(0)

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