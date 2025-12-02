"""
Data Recording Wrapper for Gymnasium Environments
This wrapper adds data recording functionality to any Gymnasium environment
using HDF5 format for efficient storage and retrieval.
"""

import os
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym

from src.dataset import HDF5DataSaver


class DataRecordingWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper that records observations and actions to HDF5 files.

    This wrapper can be applied to any Gymnasium environment to automatically
    save interaction data (observations, actions, rewards, etc.) for later
    analysis or training.

    Args:
        env: The Gymnasium environment to wrap
        save_data: Whether to actually save data (can be disabled for testing)
        output_dir: Directory to save the data files
        model_path: Path to the model being tested (for metadata)
        filename_prefix: Prefix for the output filename
    """

    def __init__(
        self,
        env: gym.Env,
        output_dir: str = "outputs/dataset",
        filename_prefix: str = "recorded_data",
    ):
        super().__init__(env)

        self.output_dir = output_dir
        self.filename_prefix = filename_prefix

        # Data recording state
        self.data_saver = None
        self.step_count = 0
        self.current_episode_steps = 0

        # Initialize data saver if saving is enabled

        self._initialize_data_saver()

    def _initialize_data_saver(self):
        """Initialize the HDF5 data saver"""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filename_prefix}_{timestamp}.h5"
        data_filepath = os.path.join(self.output_dir, filename)

        print(f"Data will be saved to: {data_filepath}")
        self.data_saver = HDF5DataSaver(data_filepath)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:

        # Reset the wrapped environment
        observation, info = self.env.reset(seed=seed, options=options)

        # Reset episode tracking
        self.current_episode_steps = 0
        self.observation = observation

        return observation, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute action and record the interaction data"""
        # Execute the action in the wrapped environment
        next_observation, reward, terminated, truncated, info = self.env.step(action)

        # Update tracking variables
        self.step_count += 1
        self.current_episode_steps += 1

        # Save the data if recording is enabled
        if self.data_saver:
            self._save_step_data(self.observation, action)
        self.observation = next_observation

        return next_observation, reward, terminated, truncated, info

    def _save_step_data(
        self,
        observation: Any,
        action: Any,
    ):
        """Save data for a single step"""
        # Convert observation to numpy array if needed
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        # Convert action to numpy array if needed
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        # Save the step data
        self.data_saver.save_step_data(
            step=self.current_episode_steps,
            observation=observation,
            action=action,
        )

    def _finalize_episode(self):
        """Finalize the current episode data"""
        if self.data_saver:
            self.data_saver.finalize()

    def close(self):
        self._finalize_episode()
        print(f"Data recording completed. Total steps recorded: {self.step_count}")

        super().close()
