import os
import random
import argparse
import numpy as np
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic for LunarLander-v2
    Input: state vector
    Outputs:
        - policy logits
        - state value
    """
    def __init__(self, state_dim, num_actions):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.shared(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value

    def act(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(self, states, actions):
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


class PPOAgent:
    def __init__(self, env_name="LunarLander-v2", args=None):
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)

        self.state_dim = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.model = ActorCritic(self.state_dim, self.num_actions).to(self.device)
        self.model.apply(init_weights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_eps = args.clip_eps
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        self.max_grad_norm = args.max_grad_norm

        self.rollout_steps = args.rollout_steps
        self.update_epochs = args.update_epochs
        self.mini_batch_size = args.mini_batch_size
        self.max_episode_steps = args.max_episode_steps
        self.total_updates = args.total_updates
        self.eval_every = args.eval_every
        self.reward_threshold = args.reward_threshold

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.history_path = os.path.join(self.save_dir, "training_history.npy")
        self.summary_path = os.path.join(self.save_dir, "training_summary.npy")

        self.history = {
            "update": [],
            "train_episode_reward": [],
            "eval_update": [],
            "eval_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

        self.best_reward = -float("inf")
        self.threshold_update = None

        self.env_steps = 0
        self.buffer = RolloutBuffer()

    def collect_rollout(self):
        self.buffer.clear()
        episode_rewards = []

        obs, _ = self.env.reset()
        state = np.array(obs, dtype=np.float32)
        ep_reward = 0.0
        ep_step = 0

        for _ in range(self.rollout_steps):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, entropy, value = self.model.act(state_tensor)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            self.buffer.states.append(state)
            self.buffer.actions.append(action.item())
            self.buffer.log_probs.append(log_prob.item())
            self.buffer.rewards.append(reward)
            self.buffer.dones.append(done)
            self.buffer.values.append(value.item())

            state = np.array(next_obs, dtype=np.float32)
            ep_reward += reward
            ep_step += 1
            self.env_steps += 1

            if done or ep_step >= self.max_episode_steps:
                episode_rewards.append(ep_reward)
                obs, _ = self.env.reset()
                state = np.array(obs, dtype=np.float32)
                ep_reward = 0.0
                ep_step = 0

        with torch.no_grad():
            final_state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            _, final_value = self.model.forward(final_state)
            next_value = final_value.item()

        return episode_rewards, next_value

    def compute_gae(self, next_value):
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values + [next_value]

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.buffer.values)]

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, advantages, returns):
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)

        n = states.size(0)
        indices = np.arange(n)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs, entropy, values = self.model.evaluate_actions(mb_states, mb_actions)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()
                last_entropy = entropy_loss.item()

        return last_policy_loss, last_value_loss, last_entropy

    def evaluate(self, episodes=5):
        rewards = []

        for _ in range(episodes):
            obs, _ = self.test_env.reset()
            state = np.array(obs, dtype=np.float32)
            done = False
            ep_reward = 0.0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits, _ = self.model.forward(state_tensor)
                    action = torch.argmax(logits, dim=1).item()

                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                state = np.array(next_obs, dtype=np.float32)

                ep_reward += reward
                step_count += 1

            rewards.append(ep_reward)

        return float(np.mean(rewards))

    def save_history(self):
        np.save(self.history_path, self.history, allow_pickle=True)

        summary = {
            "best_eval_reward": self.best_reward,
            "final_eval_reward": self.history["eval_reward"][-1] if len(self.history["eval_reward"]) > 0 else None,
            "mean_eval_reward": float(np.mean(self.history["eval_reward"])) if len(self.history["eval_reward"]) > 0 else None,
            "threshold_update": self.threshold_update,
            "total_env_steps": self.env_steps,
        }
        np.save(self.summary_path, summary, allow_pickle=True)

    def run(self):
        for update_idx in range(1, self.total_updates + 1):
            episode_rewards, next_value = self.collect_rollout()
            advantages, returns = self.compute_gae(next_value)
            policy_loss, value_loss, entropy = self.update(advantages, returns)

            self.history["update"].append(update_idx)
            self.history["policy_loss"].append(policy_loss)
            self.history["value_loss"].append(value_loss)
            self.history["entropy"].append(entropy)

            if len(episode_rewards) > 0:
                self.history["train_episode_reward"].append(float(np.mean(episode_rewards)))
            else:
                self.history["train_episode_reward"].append(np.nan)

            print(
                f"[Update {update_idx}/{self.total_updates}] "
                f"Mean Train Reward: {self.history['train_episode_reward'][-1]:.2f} | "
                f"Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | "
                f"Entropy: {entropy:.4f} | Env Steps: {self.env_steps}"
            )

            if update_idx % self.eval_every == 0:
                eval_reward = self.evaluate(episodes=5)

                self.history["eval_update"].append(update_idx)
                self.history["eval_reward"].append(eval_reward)

                if self.reward_threshold is not None and self.threshold_update is None:
                    if eval_reward >= self.reward_threshold:
                        self.threshold_update = update_idx

                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    best_model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"Saved new best model to {best_model_path} with reward {eval_reward:.2f}")

                self.save_history()

                print(
                    f"[Eval] Update: {update_idx} | "
                    f"Eval Reward: {eval_reward:.2f} | Best: {self.best_reward:.2f}"
                )

        self.save_history()
        final_model_path = os.path.join(self.save_dir, "final_model.pt")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"[Info] Final model saved to: {final_model_path}")
        print(f"[Info] History saved to: {self.history_path}")
        print(f"[Info] Summary saved to: {self.summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument("--save-dir", type=str, default="/workspace/535505/ass3/save_model/ppo_results/efc005")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)

    parser.add_argument("--reward-threshold", type=float, default=200.0)

    args = parser.parse_args()

    set_seed(42)

    agent = PPOAgent(env_name=args.env_name, args=args)
    agent.run()