from brax import jumpy as jp
from brax.envs import env
from .humanoid_mimic import HumanoidMimic
from .losses import *
import jax


class HumanoidMimicTrain(HumanoidMimic):
    """Trains a humanoid to mimic reference motion."""

    def __init__(self, system_config, reference_traj, cycle_len,
                 early_termination, demo_replay_mode, err_threshold, replay_rate, reward_scaling=1.0, vel_weight=0.):
        super().__init__(system_config, reference_traj, cycle_len, reward_scaling, vel_weight)
        self.early_termination = early_termination
        self.demo_replay_mode = demo_replay_mode
        self.err_threshold = err_threshold
        self.replay_rate = replay_rate

    def reset(self, rng: jp.ndarray) -> env.State:
        replay_key, rng = jp.random_split(rng)
        state = super(HumanoidMimicTrain, self).reset(rng)
        state.metrics.update(replay=jp.zeros(1)[0])
        state.metrics.update(replay_key=rng)
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super(HumanoidMimicTrain, self).step(state, action)
        if self.early_termination:
            state = state.replace(done=state.metrics['fall'])
        state = self._demo_replay(state)
        return state

    def _demo_replay(self, state) -> env.State:
        qp = state.qp
        ref_qp = self._get_ref_state(state.metrics['step_index'])
        if self.demo_replay_mode == 'threshold':
            error = loss_l1_pos(qp, ref_qp)
            replay = jp.where(error > self.err_threshold, jp.float32(1), jp.float32(0))
        elif self.demo_replay_mode == 'random':
            replay_key, key = jax.random.split(state.metrics['replay_key'])
            state.metrics.update(replay_key=replay_key)
            replay = jp.where(jax.random.bernoulli(key, p=self.replay_rate), jp.float32(1), jp.float32(0))
        else:
            raise NotImplementedError
        qp = jp.tree_map(lambda x: x*(1 - replay), qp) + jp.tree_map(lambda x: x*replay, ref_qp)
        obs = self._get_obs(qp, state.metrics['step_index'])
        state.metrics.update(replay=replay)
        return state.replace(qp=qp, obs=obs)
