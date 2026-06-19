from __future__ import annotations

from isaaclab.utils import configclass

from whole_body_tracking.rl.fast_sac.config import FastSACAgentCfg
from whole_body_tracking.rl.fast_sac.recipe_contract import FAST_SAC_RECIPE_CONTRACT_ID, FAST_SAC_RECIPE_CONTRACT_VERSION
from whole_body_tracking.tasks.tracking.config.g1.paper_contract import (
    G1_PAPER_EQUIVALENCE_CONTRACT_ID,
    G1_PAPER_EQUIVALENCE_CONTRACT_VERSION,
)


@configclass
class G1PaperFastSACRunnerCfg(FastSACAgentCfg):
    """FastSAC config for paper-full G1 1307 training."""

    experiment_name = "g1_paper_fast_sac"
    run_name = ""
    max_steps = 30_000
    warmup_steps = 10  # paper-aligned (was 10_000)
    updates_per_step = 2  # paper-aligned UTD=2 (was 4)
    replay_size = 204_800  # FastTD3-style short buffer ~50 steps x 4096 envs (was 1_000_000)
    batch_size = 32_768  # paper-aligned FastSAC reference default (was 256)
    tau = 0.1  # paper-aligned FastSAC default (was 0.005)
    critic_num_atoms = 101  # paper-aligned (was 51)
    critic_v_min = -250.0  # paper-aligned symmetric reference support
    critic_v_max = 250.0
    actor_hidden_dims = [512, 512, 512]  # paper ref: 3x512
    critic_hidden_dims = [1024, 1024, 1024, 1024]  # paper ref: 4x1024
    source_note = "Paper-only FastSAC path for arXiv:2602.13656 with arXiv:2512.01996 recipe defaults."
    paper_equivalence_contract_id = G1_PAPER_EQUIVALENCE_CONTRACT_ID
    paper_equivalence_contract_version = G1_PAPER_EQUIVALENCE_CONTRACT_VERSION
    fast_sac_recipe_contract_id = FAST_SAC_RECIPE_CONTRACT_ID
    fast_sac_recipe_contract_version = FAST_SAC_RECIPE_CONTRACT_VERSION
