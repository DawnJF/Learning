import os
import sys
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import logging
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from envs.data_recording_wrapper import DataRecordingWrapper
from src.sac_policy.low_dimensional import Actor, SoftQNetwork
from src.buffers import ReplayBuffer
from src.utils import get_device, logging_args, setup_logging


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).split(".")[0]
    """the name of this experiment"""
    output_dir: str = "outputs"
    render: bool = False

    # Algorithm specific arguments
    # env_id: str = "HalfCheetah-v4"
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    load_checkpoint: str = None
    """path to load checkpoint from"""
    learning_starts: int = 5000
    """timestep to start learning"""
    total_timesteps: int = 80_000
    """total timesteps of the experiments"""
    save_freq: int = 10_000
    """frequency to save checkpoints"""

    gamma: float = 0.99
    """the discount factor gamma"""

    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""

    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, render=False):
    env = gym.make(env_id, render_mode="human" if render else None)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = DataRecordingWrapper(
    #     env,
    #     output_dir="outputs/dataset/Hopper-v4/2025-12-03-16-25-33_79999",
    # )
    return env


def save_checkpoint(
    actor, qf, actor_optimizer, q_optimizer, global_step, args, best=False
):
    """Save model checkpoint"""
    checkpoint_path = args.output_dir

    checkpoint = {
        "global_step": global_step,
        "actor_state_dict": actor.state_dict(),
        "qf_state_dict": qf.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "args": args.__dict__,
    }

    file_name = (
        f"{checkpoint_path}/checkpoint_best.pt"
        if best
        else f"{checkpoint_path}/checkpoint_{global_step}.pt"
    )

    torch.save(checkpoint, file_name)
    logging.info(f"Checkpoint saved at step {global_step}, to {file_name}")


def load_checkpoint(checkpoint_path, actor, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    # qf1.load_state_dict(checkpoint["qf1_state_dict"])
    # qf2.load_state_dict(checkpoint["qf2_state_dict"])
    # actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    # q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])

    return checkpoint["global_step"], checkpoint["args"]


def evaluate_agent(load_checkpoint_path, num_episodes=5, max_steps=4000, render=False):
    """Evaluate the agent and optionally render the environment"""

    args = Args()

    env = make_env(args.env_id, render=render)
    action_space = env.action_space
    observation_space = env.observation_space
    print(f"action_space: {action_space}")
    print(f"observation_space: {observation_space}")

    actor = Actor(observation_space, action_space)

    load_checkpoint(load_checkpoint_path, actor, device="cpu")

    episode_returns = []
    episode_lengths = []

    for episode in range(num_episodes):
        print(" " * 40)
        print(f"Starting evaluation episode {episode + 1}/{num_episodes}")
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                _, _, action = actor.get_action(obs_tensor)
                action = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)

            reward_scalar = np.asarray(reward).item()
            episode_return += reward_scalar
            episode_length += 1
            done = terminated or truncated

            if done:
                returns = info["episode"]["r"]
                length = info["episode"]["l"]
                print(f"Returns: {returns:.2f}, Length: {length}")
                print("terminated:", terminated, "truncated:", truncated)

            # if render:
            # eval_env.render()
            # time.sleep(0.01)  # Small delay for better visualization

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    env.close()
    print(" " * 40)

    avg_return = np.mean(episode_returns)
    avg_length = np.mean(episode_lengths)

    print(
        f"Evaluation Results: Avg Return = {float(avg_return):.2f}, Avg Length = {float(avg_length):.2f}"
    )
    return avg_return, avg_length


def train():

    args = tyro.cli(Args)

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"{args.env_id}__{args.exp_name}__{time_str}"
    args.output_dir = os.path.join(args.output_dir, args.env_id, run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    setup_logging(args.output_dir)

    logging_args(args)
    writer = SummaryWriter(args.output_dir)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = make_env(args.env_id, args.render)

    action_space = env.action_space
    observation_space = env.observation_space

    logging.info("=" * 40)
    logging.info(f"action_space: {action_space}")
    logging.info(f"observation_space: {observation_space}")
    logging.info("=" * 40)

    actor = Actor(observation_space, action_space).to(device)
    qf = SoftQNetwork(observation_space, action_space).to(device)
    logging.info(actor)
    logging.info(qf)

    qf_target = SoftQNetwork(observation_space, action_space).to(device)
    qf_target.load_state_dict(qf.state_dict())
    q_optimizer = optim.AdamW(list(qf.parameters()), lr=args.q_lr)
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.AdamW([log_alpha], lr=args.q_lr)
        logging.info(f"Target entropy set to {target_entropy:.2f}")
    else:
        alpha = args.alpha

    observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        observation_space,
        action_space,
        device,
        n_envs=1,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    best_avg_return = -np.inf

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset()

    for global_step in range(args.total_timesteps):

        if global_step == args.learning_starts:
            logging.info("Learning starts now!")

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action_sample = action_space.sample()
            actions = np.array([action_sample])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).unsqueeze(0).to(device))
            actions = actions.detach().cpu().numpy()

        # execute the game
        action_to_step = actions.squeeze() if len(actions.shape) > 1 else actions
        next_obs, reward, terminated, truncated, info = env.step(action_to_step)

        # TRY NOT TO MODIFY: record reward for plotting purposes
        returns = 0
        if terminated or truncated:
            if "episode" in info:
                returns = info["episode"]["r"]
                length = info["episode"]["l"]
                logging.info(
                    f"global_step={global_step}, returns={returns:.2f}, length={length}"
                )
                writer.add_scalar("charts/episodic_return", returns, global_step)
                writer.add_scalar("charts/episodic_length", length, global_step)

        # 确保数据格式正确用于单环境
        rb.add(
            obs.reshape(1, -1),
            next_obs.reshape(1, -1),
            actions.reshape(1, -1),
            np.array([reward]).astype(np.float32),
            np.array([terminated]).astype(np.float32),
            [info],
        )

        # if returns > best_avg_return and global_step > args.save_freq:
        #     best_avg_return = returns
        #     logging.info(
        #         f"New best average return: {best_avg_return:.2f} at step {global_step}"
        #     )
        #     save_checkpoint(
        #         actor, qf, actor_optimizer, q_optimizer, global_step, args, True
        #     )

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue

        data = rb.sample(args.batch_size)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(
                data.next_observations
            )
            # [B, N]
            qf_next_target = qf_target(data.next_observations, next_state_actions)
            # qf_next_target = torch.topk(qf_next_target, k=2, dim=1)[0][:, -1]
            qf_next_target = torch.min(qf_next_target, dim=1)[0]

            qf_next_target = qf_next_target - alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (
                1 - data.dones.flatten()
            ) * args.gamma * (qf_next_target).view(-1)

        qf_a_values = qf(data.observations, data.actions)
        qf_loss = (
            F.mse_loss(qf_a_values[:, 0], next_q_value)
            + F.mse_loss(qf_a_values[:, 1], next_q_value)
        ) / 2.0

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
            # compensate for the delay by doing 'actor_update_interval' instead of 1
            for _ in range(args.policy_frequency):

                # data = rb.sample(args.batch_size)
                pi, log_pi, _ = actor.get_action(data.observations)
                qf_pi = qf(data.observations, pi)
                qf_pi = torch.mean(qf_pi, dim=1)

                actor_loss = ((alpha * log_pi) - qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        # update the target networks
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

        if global_step % 100 == 0:
            writer.add_scalar(
                "charts/qf1_values", qf_a_values[:, 0].mean().item(), global_step
            )
            writer.add_scalar(
                "charts/qf2_values", qf_a_values[:, 1].mean().item(), global_step
            )
            writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("charts/alpha", alpha, global_step)
            writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            writer.add_scalar("charts/entropy", -log_pi.mean().item(), global_step)
            writer.add_scalar(
                "charts/SPS",
                int(global_step / (time.time() - start_time)),
                global_step,
            )
        if global_step % 500 == 0:
            logging.info(
                f"Step: {global_step}, "
                f"QF Loss: {qf_loss.item():.3f}, "
                f"Actor Loss: {actor_loss.item():.3f}, "
                f"Alpha: {alpha:.3f}, "
                f"Entropy: {-log_pi.mean().item():.3f}, "
                f"SPS: {int(global_step / (time.time() - start_time))}"
            )
        # Save checkpoint
        if global_step % args.save_freq == 0:
            save_checkpoint(actor, qf, actor_optimizer, q_optimizer, global_step, args)

    # Final checkpoint save
    save_checkpoint(actor, qf, actor_optimizer, q_optimizer, global_step, args)

    env.close()
    writer.close()


def test():
    """
    双q共享可训练特征层，前期效果好，因为"容易高估"，但后期不够稳定
    独立Q 后期更稳定，但是更慢，整体来看 训练更稳定

    AdamW 更好

    每次loss sample data 好像影响不大

    actor 用 mean 收敛快

    结论：AdamW + actor mean q + （独立双Q）
    """
    path = (
        "outputs/sac/Hopper-v4__sac__2025-10-24-11-55-54/checkpoint_60000.pt"  # Q共享
    )
    path = (
        "outputs/sac/Hopper-v4__sac__2025-10-24-14-32-24/checkpoint_60000.pt"  # Q独立
    )

    path = "outputs/sac/Hopper-v4__sac__2025-10-24-15-20-27/checkpoint_60000.pt"

    """
    结论：discount 影响巨大... 
    0.999 会快非常多; 0.97 会差非常多，一直学的不好
    """

    path = "outputs/sac/Hopper-v4__q_discount_999__2025-10-27-16-25-52/checkpoint_120000.pt"
    path = "outputs/sac/HalfCheetah-v4__q_discount_999__2025-11-17-10-23-21/checkpoint_99999.pt"

    """
    # 12.3
    .....发现0.999其实是 reward hack 了，一直原地不动, 0.99没问题

    """
    evaluate_agent(path, 5)


def test_render():

    path = "outputs/sac/Hopper-v4__099__2025-12-03-16-25-33/checkpoint_79999.pt"

    evaluate_agent(path, 2, render=True)


if __name__ == "__main__":
    # test()
    # train()
    test_render()
