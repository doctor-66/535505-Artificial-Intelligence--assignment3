import os
import random
import argparse
from collections import deque
import cv2
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class AtariPreprocessor:
    """
    Preprocess Atari RGB frames into stacked grayscale frames.
    - grayscale
    - resize to 84x84
    - stack 4 frames
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0).astype(np.float32) / 255.0

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0).astype(np.float32) / 255.0


class DQN(nn.Module):
    """
    CNN-based DQN for Atari.
    Input shape: (B, 4, 84, 84)
    Output shape: (B, num_actions)
    """
    def __init__(self, num_actions):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, frameskip=1, render_mode="rgb_array")
        self.test_env = gym.make(env_name, frameskip=1, render_mode="rgb_array")

        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor(frame_stack=args.frame_stack)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)

        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.memory = ReplayBuffer(args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -float("inf")

        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir

        self.history = {
            "train_episode": [],
            "train_reward": [],
            "eval_episode": [],
            "eval_env_step": [],
            "eval_reward": [],
        }
        self.reward_threshold = args.reward_threshold
        self.threshold_episode = None
        self.threshold_env_step = None

        os.makedirs(self.save_dir, exist_ok=True)
        self.history_path = os.path.join(self.save_dir, "training_history.npy")
        self.summary_path = os.path.join(self.save_dir, "training_summary.npy")
        

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax(dim=1).item()

    def save_history(self):
        np.save(
            self.history_path,
            self.history,
            allow_pickle=True
        )

        summary = {
            "best_eval_reward": self.best_reward,
            "final_eval_reward": self.history["eval_reward"][-1] if len(self.history["eval_reward"]) > 0 else None,
            "mean_eval_reward": float(np.mean(self.history["eval_reward"])) if len(self.history["eval_reward"]) > 0 else None,
            "threshold_episode": self.threshold_episode,
            "threshold_env_step": self.threshold_env_step,
            "total_env_steps": self.env_count,
            "total_updates": self.train_count,
        }

        np.save(self.summary_path,summary,allow_pickle=True)

    def run(self, episodes=500):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)

            done = False
            total_reward = 0.0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)
                self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:
                    print(
                        f"[Collect] Ep: {ep} Step: {step_count} "
                        f"SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}"
                    )
            
            self.history["train_episode"].append(ep)
            self.history["train_reward"].append(total_reward)
            print(
                f"[Episode] Ep: {ep} Total Reward: {total_reward:.2f} "
                f"SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}"
            )

            if ep % 50 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                self.history["eval_episode"].append(ep)
                self.history["eval_env_step"].append(self.env_count)
                self.history["eval_reward"].append(eval_reward)

                if self.reward_threshold is not None and self.threshold_episode is None:
                    if eval_reward >= self.reward_threshold:
                        self.threshold_episode = ep
                        self.threshold_env_step = self.env_count

                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward:.2f}")
                
                self.save_history()
                
                print(
                    f"[Eval] Ep: {ep} Eval Reward: {eval_reward:.2f} "
                    f"SC: {self.env_count} UC: {self.train_count}"
                )
        
        print(f"[Info] History saved to: {self.history_path}")
        print(f"[Info] Summary saved to: {self.summary_path}")

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0.0
        step_count = 0

        while not done and step_count < self.max_episode_steps:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)
            step_count += 1

        return total_reward

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.train_count += 1

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(
                f"[Train #{self.train_count}] Loss: {loss.item():.4f} "
                f"Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5")
    parser.add_argument("--save-dir", type=str, default="/workspace/535505/ass3/save_model/atari_results/bs_3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99999)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=5000)
    parser.add_argument("--replay-start-size", type=int, default=100000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--reward-threshold", type=float, default=18.0)
    args = parser.parse_args()

    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run(episodes=args.episodes)