# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
import random
import time
import logging
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from src.utils import logging_args, setup_logging


@dataclass
class Args:
    file_name: str = os.path.basename(__file__).split(".")[0]
    """the name of this experiment"""
    experiment_tag: str = "no_R_normalize"
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    output_dir: str = "outputs"

    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    # env_id: str = "HalfCheetah-v4"
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 500000
    gamma: float = 0.99
    """the discount factor gamma"""

    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""

    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Checkpoint arguments
    save_interval: int = 10
    """save checkpoint every N iterations"""
    log_interval: int = 20
    """log training metrics every N iterations"""
    resume_from: str = None
    """path to checkpoint to resume training from"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, gamma, render=False):
    def thunk():
        if render and idx == 0:
            # env = gym.make(env_id, render_mode="rgb_array")
            env = gym.make(env_id, render_mode="human")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

    def act(self, x):
        action_mean = self.actor_mean(torch.Tensor(x))
        return action_mean.detach().cpu().numpy()


def save_checkpoint(
    save_path,
    agent,
    optimizer,
    iteration,
    global_step,
    best_episodic_return,
    args=None,
    **kwargs,
):
    """
    Save training checkpoint

    Args:
        save_path: Path to save checkpoint
        agent: Agent model
        optimizer: Optimizer
        iteration: Current iteration
        global_step: Current global step
        best_episodic_return: Best episodic return achieved
        args: Training arguments (optional)
        **kwargs: Additional information to save
    """
    checkpoint = {
        "iteration": iteration,
        "global_step": global_step,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_episodic_return": best_episodic_return,
    }

    if args is not None:
        checkpoint["args"] = vars(args) if not isinstance(args, dict) else args

    # Add any additional kwargs
    checkpoint.update(kwargs)

    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to: {save_path}")

    return save_path


def load_checkpoint(checkpoint_path, agent, optimizer=None, device="cpu"):
    """
    Load training checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        agent: Agent model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load tensors to

    Returns:
        dict: Checkpoint information including iteration, global_step, etc.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model state
    agent.load_state_dict(checkpoint["model_state_dict"])
    logging.info("Model state loaded successfully")

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logging.info("Optimizer state loaded successfully")

    # Extract checkpoint info
    checkpoint_info = {
        "iteration": checkpoint.get("iteration", 0),
        "global_step": checkpoint.get("global_step", 0),
        "best_episodic_return": checkpoint.get("best_episodic_return", float("-inf")),
        "args": checkpoint.get("args", None),
    }

    logging.info(
        f"Checkpoint info: iteration={checkpoint_info['iteration']}, "
        f"global_step={checkpoint_info['global_step']}, "
        f"best_return={checkpoint_info['best_episodic_return']:.2f}"
    )

    return checkpoint_info


def train():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    time_str = time.strftime("%y-%m%d-%H-%M-%S")
    run_name = f"ppo_{time_str}"
    if args.experiment_tag:
        run_name += f"_{args.experiment_tag}"
    args.output_dir = os.path.join(
        args.output_dir, args.env_id, args.file_name, run_name
    )
    os.makedirs(args.output_dir, exist_ok=True)

    setup_logging(args.output_dir)
    logging_args(args)

    logging.info(f"Output directory: {args.output_dir}")

    writer = SummaryWriter(args.output_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"Using device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    num_params = sum(p.numel() for p in agent.parameters())
    num_trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    logging.info(
        f"Agent created with {num_params:,} parameters ({num_trainable_params:,} trainable)"
    )

    # Load checkpoint if resume_from is specified
    start_iteration = 1
    start_global_step = 0
    best_episodic_return = float("-inf")
    episode_returns = []

    if args.resume_from is not None:
        checkpoint_info = load_checkpoint(args.resume_from, agent, optimizer, device)
        start_iteration = checkpoint_info["iteration"] + 1
        start_global_step = checkpoint_info["global_step"]
        best_episodic_return = checkpoint_info["best_episodic_return"]
        logging.info(
            f"Resuming training from iteration {start_iteration}, global step {start_global_step}"
        )
        logging.info(f"Best episodic return so far: {best_episodic_return:.2f}")

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = start_global_step
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    logging.info("Starting training loop...")

    for iteration in tqdm(range(start_iteration, args.num_iterations + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "episode" in infos:
                for i in range(args.num_envs):
                    episodic_return = infos["episode"]["r"][i]
                    episodic_length = infos["episode"]["l"][i]
                    episode_returns.append(episodic_return)

                    writer.add_scalar(
                        "charts/episodic_return", episodic_return, global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", episodic_length, global_step
                    )

                    # Save best model
                    if episodic_return > best_episodic_return:
                        best_episodic_return = episodic_return
                        best_model_path = os.path.join(
                            args.output_dir, "best_model.pth"
                        )
                        save_checkpoint(
                            best_model_path,
                            agent,
                            optimizer,
                            iteration,
                            global_step,
                            best_episodic_return,
                            args,
                        )
                        logging.info(
                            f"New best model saved! Return: {best_episodic_return:.2f}"
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

        # Periodic logging
        if iteration % args.log_interval == 0:
            avg_return = (
                np.mean(episode_returns[-10:]) if len(episode_returns) > 0 else 0
            )
            logging.info("=" * 80)
            logging.info(
                f"Iteration {iteration}/{args.num_iterations} | Global Step: {global_step:,}"
            )
            logging.info(
                f"  SPS: {sps} | Time Elapsed: {time.time() - start_time:.1f}s"
            )
            logging.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            logging.info(
                f"  Policy Loss: {pg_loss.item():.4f} | Value Loss: {v_loss.item():.4f}"
            )
            logging.info(
                f"  Entropy: {entropy_loss.item():.4f} | Approx KL: {approx_kl.item():.4f}"
            )
            logging.info(
                f"  Clip Frac: {np.mean(clipfracs):.4f} | Explained Var: {explained_var:.4f}"
            )
            logging.info(
                f"  Avg Return (last 10 eps): {avg_return:.2f} | Best Return: {best_episodic_return:.2f}"
            )
            logging.info("=" * 80)

        # Periodic checkpoint saving
        if iteration % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint_iter_{iteration}.pth"
            )
            save_checkpoint(
                checkpoint_path,
                agent,
                optimizer,
                iteration,
                global_step,
                best_episodic_return,
                args,
                episode_returns=episode_returns,
            )

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    save_checkpoint(
        final_model_path,
        agent,
        optimizer,
        args.num_iterations,
        global_step,
        best_episodic_return,
        args,
        total_episodes=len(episode_returns),
        episode_returns=episode_returns,
    )

    # Also save just the model state dict for compatibility
    legacy_model_path = os.path.join(args.output_dir, "final.pth")
    torch.save(agent.state_dict(), legacy_model_path)

    total_time = time.time() - start_time
    logging.info("=" * 80)
    logging.info("Training completed!")
    logging.info(f"Total training time: {total_time/3600:.2f} hours")
    logging.info(f"Total episodes: {len(episode_returns)}")
    logging.info(f"Best episodic return: {best_episodic_return:.2f}")
    if len(episode_returns) > 0:
        logging.info(
            f"Average return (last 100 eps): {np.mean(episode_returns[-100:]):.2f}"
        )
        logging.info(
            f"Final return (last 10 eps): {np.mean(episode_returns[-10:]):.2f}"
        )
    logging.info(f"Final model saved to: {final_model_path}")
    logging.info(f"Legacy model saved to: {legacy_model_path}")
    logging.info(
        f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pth')}"
    )
    logging.info("=" * 80)

    print(f"model saved to {final_model_path}")

    envs.close()
    writer.close()


def eval():
    path = "runs/HalfCheetah-v4__ppo__1__1762329773/ppo.cleanrl_model"
    path = "runs/Hopper-v4__ppo__1__1762336668/ppo.cleanrl_model"
    path = "outputs/ppo/Hopper-v4__ppo__2025-11-21-15-29-15/checkpoint_iter_900.pth"
    path = "outputs/ppo_test/Hopper-v4__ppo__2025-11-27-11-52-16/best_model.pth"
    path = "outputs/Hopper-v4/ppo/Hopper-v4__ppo__2025-12-03-17-39-07/checkpoint_iter_40.pth"

    args = Args()

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.gamma, render=True)
            for i in range(args.num_envs)
        ]
    )

    agent = Agent(envs)
    info = load_checkpoint(path, agent)
    print(f"loaded model from {path}, info: {info}")
    agent.eval()

    obs, _ = envs.reset()

    while True:
        action = agent.act(obs)
        # time.sleep(0.04)
        obs, reward, terminations, truncations, infos = envs.step(action)
        if "episode" in infos:
            for i in range(args.num_envs):
                episodic_return = infos["episode"]["r"][i]
                print(f"episodic_return: {episodic_return:.2f}")
        if terminations[0] or truncations[0]:
            obs, _ = envs.reset()
            print("env reset")


if __name__ == "__main__":
    train()
    # eval()
