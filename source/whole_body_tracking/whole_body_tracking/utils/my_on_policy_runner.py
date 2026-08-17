from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class MyOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        # W&B should receive scalar metrics only; do not upload repository diffs.
        self.git_status_repos = []

    def add_git_repo_to_log(self, repo_file_path):
        """Keep code snapshots out of the scalar-only W&B run."""
        return None

    def save(self, path: str, infos=None):
        """Save the checkpoint locally without uploading files to W&B."""
        previous_disable_logs = self.disable_logs
        self.disable_logs = True
        try:
            super().save(path, infos)
        finally:
            self.disable_logs = previous_disable_logs


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        # W&B should receive scalar metrics only; do not upload repository diffs.
        self.git_status_repos = []

    def add_git_repo_to_log(self, repo_file_path):
        """Keep code snapshots out of the scalar-only W&B run."""
        return None

    def save(self, path: str, infos=None):
        """Save the checkpoint locally without uploading files to W&B."""
        previous_disable_logs = self.disable_logs
        self.disable_logs = True
        try:
            super().save(path, infos)
        finally:
            self.disable_logs = previous_disable_logs
