import h5py
import numpy as np
import os
from datetime import datetime
import threading
import queue
from typing import Optional, Tuple, Any


class HDF5DataSaver:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.step_count = 0
        self.datasets = {}

        # Threading components
        self.data_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Open HDF5 file and initialize datasets
        self._initialize_file()

        # Start worker thread
        self._start_worker_thread()

    def _initialize_file(self):
        """Initialize HDF5 file and datasets"""
        self.file = h5py.File(self.filepath, "w")

        # Save metadata
        metadata_group = self.file.create_group("metadata")
        metadata_group.attrs["start_time"] = datetime.now().isoformat()

        # Create datasets with unlimited size along first dimension
        # We'll determine the observation shape from the first data point
        self.shape_initialized = False

        # Create step numbers dataset
        self.datasets["steps"] = self.file.create_dataset(
            "steps", (0,), maxshape=(None,), dtype=np.int32, chunks=True
        )

    def _start_worker_thread(self):
        """Start the background worker thread"""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """Background worker thread that handles actual HDF5 writing"""
        while not self.stop_event.is_set():
            try:
                # Wait for data with timeout to allow checking stop_event
                data_item = self.data_queue.get(timeout=1.0)
                if data_item is None:  # Sentinel value to stop
                    break

                step, observation, action = data_item
                self._write_step_data(step, observation, action)
                self.data_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker thread: {e}")
                continue

    def _write_step_data(self, step: int, observation: np.ndarray, action: np.ndarray):
        """Actually write data to HDF5 (runs in background thread)"""
        if not self.shape_initialized:
            # Initialize observations dataset with proper shape
            obs_shape = observation.shape

            self.datasets["observations"] = self.file.create_dataset(
                "observations",
                (0,) + obs_shape,
                maxshape=(None,) + obs_shape,
                dtype=observation.dtype,
                chunks=True,
            )
            action_shape = action.shape
            self.datasets["actions"] = self.file.create_dataset(
                "actions",
                (0,) + action_shape,
                maxshape=(None,) + action_shape,
                dtype=action.dtype,
                chunks=True,
            )
            self.shape_initialized = True

        # Resize datasets to accommodate new data
        self.datasets["steps"].resize((self.step_count + 1,))
        self.datasets["actions"].resize(
            (self.step_count + 1,) + self.datasets["actions"].shape[1:]
        )
        self.datasets["observations"].resize(
            (self.step_count + 1,) + self.datasets["observations"].shape[1:]
        )

        # Save data
        self.datasets["steps"][self.step_count] = step
        self.datasets["actions"][self.step_count] = action
        self.datasets["observations"][self.step_count] = observation

        self.step_count += 1

        # Flush to disk periodically for safety
        if self.step_count % 100 == 0:
            self.file.flush()

    def save_step_data(self, step: int, observation: np.ndarray, action: np.ndarray):
        """Save data for one step (non-blocking)"""
        # Just add to queue, don't block
        self.data_queue.put((step, observation.copy(), action.copy()))

    def finalize(self, extra_info: Optional[dict] = None):
        """Finalize the file with summary information"""
        # Wait for all queued data to be processed
        self.data_queue.join()

        # Signal worker thread to stop
        self.stop_event.set()
        self.data_queue.put(None)  # Sentinel value

        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join()

        if self.file:
            # Add final metadata
            self.file.attrs["end_time"] = datetime.now().isoformat()
            if extra_info:
                for key, value in extra_info.items():
                    self.file.attrs[key] = value

            # Final flush and close
            self.file.flush()
            self.file.close()
            self.file = None

    def __del__(self):
        """Ensure file is closed and thread is stopped"""
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        if (
            hasattr(self, "worker_thread")
            and self.worker_thread
            and self.worker_thread.is_alive()
        ):
            self.worker_thread.join(timeout=5.0)
        if self.file:
            self.file.close()


def load_data(filepath: str):
    """Load data from HDF5 file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = {}
    with h5py.File(filepath, "r") as file:
        # Load datasets
        data["observations"] = np.array(file["observations"])
        data["actions"] = np.array(file["actions"])
        data["steps"] = np.array(file["steps"])

        # Load metadata
        data["metadata"] = {}
        if "metadata" in file:
            metadata_group = file["metadata"]
            for key in metadata_group.attrs.keys():
                data["metadata"][key] = metadata_group.attrs[key]

        # Load file-level attributes
        for key in file.attrs.keys():
            data["metadata"][key] = file.attrs[key]

    return data


if __name__ == "__main__":

    file = (
        "outputs/dataset_s4/observations_actions_flappy_bird_800000_20250806_003553.h5"
    )

    print(f"Loading data from {file}")
    data = load_data(file)
    print(data.keys())
    print(data["observations"].shape)
