import sys
import unittest
from types import SimpleNamespace

import instinct_rl.runners.on_policy_runner as on_policy_runner
from instinct_rl.runners.on_policy_runner import OnPolicyRunner


class DummyTensorboardWriter:
    def __init__(self, log_dir=None, flush_secs=None):
        self.log_dir = log_dir
        self.flush_secs = flush_secs
        self.scalars = []

    def add_scalar(self, key, value, step):
        self.scalars.append((key, value, step))


class DummyWandbRun:
    id = "run/1"
    name = "smoke_test"

    def __init__(self):
        self.logs = []
        self.artifacts = []

    def log(self, data, step=None):
        self.logs.append((data, step))

    def log_artifact(self, artifact, aliases=None):
        self.artifacts.append((artifact, aliases))


class DummyWandbArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path, name=None):
        self.files.append((path, name))


class OnPolicyRunnerWandbTest(unittest.TestCase):
    def test_init_writers_initializes_tensorboard_and_wandb_from_config(self):
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.cfg = {
            "logger": "tensorboard,wandb",
            "experiment_name": "terrain_shadowing",
            "run_name": "smoke_test",
            "wandb_entity": "lab",
            "wandb_mode": "offline",
        }
        runner.log_dir = "/tmp/instinct_rl_logs/run1"
        runner.writer = None
        runner.wandb_run = None
        runner.logger_names = {"tensorboard", "wandb"}
        runner.is_mp_rank_other_process = lambda: False

        wandb_run = DummyWandbRun()
        wandb_module = SimpleNamespace(init=lambda **kwargs: wandb_run, kwargs=None)

        def init_wandb(**kwargs):
            wandb_module.kwargs = kwargs
            return wandb_run

        wandb_module.init = init_wandb
        old_summary_writer = on_policy_runner.SummaryWriter
        old_wandb_module = sys.modules.get("wandb")
        sys.modules["wandb"] = wandb_module
        on_policy_runner.SummaryWriter = DummyTensorboardWriter
        try:
            runner.init_writers()
        finally:
            on_policy_runner.SummaryWriter = old_summary_writer
            if old_wandb_module is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = old_wandb_module

        self.assertIsInstance(runner.writer, DummyTensorboardWriter)
        self.assertEqual(runner.writer.log_dir, "/tmp/instinct_rl_logs/run1")
        self.assertIs(runner.wandb_run, wandb_run)
        self.assertEqual(wandb_module.kwargs["project"], "terrain_shadowing")
        self.assertEqual(wandb_module.kwargs["name"], "smoke_test")
        self.assertEqual(wandb_module.kwargs["entity"], "lab")
        self.assertEqual(wandb_module.kwargs["mode"], "offline")
        self.assertEqual(wandb_module.kwargs["dir"], "/tmp/instinct_rl_logs/run1")
        self.assertEqual(wandb_module.kwargs["config"], runner.cfg)

    def test_writer_mp_add_scalar_logs_to_wandb_when_initialized(self):
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.writer = DummyTensorboardWriter()
        runner.wandb_run = DummyWandbRun()
        runner.current_learning_iteration = 7
        runner.is_mp_rank_other_process = lambda: False

        runner.writer_mp_add_scalar("Loss/value_loss", 1.25, 7)

        self.assertEqual(runner.writer.scalars, [("Loss/value_loss", 1.25, 7)])
        self.assertEqual(runner.wandb_run.logs, [({"Loss/value_loss": 1.25}, 7)])

    def test_writer_mp_add_scalar_uses_iteration_for_non_integer_wandb_step(self):
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.writer = DummyTensorboardWriter()
        runner.wandb_run = DummyWandbRun()
        runner.current_learning_iteration = 12
        runner.is_mp_rank_other_process = lambda: False

        runner.writer_mp_add_scalar("Train/time/mean_reward", 3.5, 42.75)

        self.assertEqual(runner.writer.scalars, [("Train/time/mean_reward", 3.5, 42.75)])
        self.assertEqual(runner.wandb_run.logs, [({"Train/time/mean_reward": 3.5}, 12)])

    def test_save_logs_checkpoint_as_wandb_artifact(self):
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.cfg = {}
        runner.log_dir = "/tmp/instinct_rl_logs/run1"
        runner.current_learning_iteration = 1000
        runner.wandb_run = DummyWandbRun()
        runner.normalizers = {}
        runner.alg = SimpleNamespace(state_dict=lambda: {"policy": "state"})
        runner.is_mp_rank_other_process = lambda: False

        saved = []
        wandb_module = SimpleNamespace(Artifact=DummyWandbArtifact)
        old_torch_save = on_policy_runner.torch.save
        old_wandb_module = sys.modules.get("wandb")
        on_policy_runner.torch.save = lambda state, path: saved.append((state, path))
        sys.modules["wandb"] = wandb_module
        try:
            runner.save("/tmp/instinct_rl_logs/run1/model_1000.pt")
        finally:
            on_policy_runner.torch.save = old_torch_save
            if old_wandb_module is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = old_wandb_module

        self.assertEqual(saved[0][1], "/tmp/instinct_rl_logs/run1/model_1000.pt")
        artifact, aliases = runner.wandb_run.artifacts[0]
        self.assertEqual(artifact.type, "model")
        self.assertEqual(artifact.files, [("/tmp/instinct_rl_logs/run1/model_1000.pt", "model_1000.pt")])
        self.assertEqual(aliases, ["latest", "iter-1000"])


if __name__ == "__main__":
    unittest.main()
