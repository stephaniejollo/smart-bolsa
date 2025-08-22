
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Union, IO

@dataclass
class AgentConfig:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.999
    seed: int = 42

class DoubleQAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.Q1: Dict[int, np.ndarray] = {}
        self.Q2: Dict[int, np.ndarray] = {}
        self.visit_counts: Dict[int, int] = {}

    def _ensure_state(self, s: int):
        if s not in self.Q1:
            self.Q1[s] = np.zeros((self.cfg.n_actions,), dtype=np.float32)
            self.Q2[s] = np.zeros((self.cfg.n_actions,), dtype=np.float32)
            self.visit_counts[s] = 0

    def policy(self, state: int) -> int:
        self._ensure_state(state)
        if self.rng.random() < self.cfg.epsilon:
            return int(self.rng.integers(0, self.cfg.n_actions))
        qsum = self.Q1[state] + self.Q2[state]
        return int(np.argmax(qsum))

    def update(self, s: int, a: int, r: float, ns: int, done: bool):
        self._ensure_state(s); self._ensure_state(ns)
        if self.rng.random() < 0.5:
            if done:
                target = r
            else:
                a_star = int(np.argmax(self.Q1[ns]))
                target = r + self.cfg.gamma * self.Q2[ns][a_star]
            self.Q1[s][a] += self.cfg.alpha * (target - self.Q1[s][a])
        else:
            if done:
                target = r
            else:
                a_star = int(np.argmax(self.Q2[ns]))
                target = r + self.cfg.gamma * self.Q1[ns][a_star]
            self.Q2[s][a] += self.cfg.alpha * (target - self.Q2[s][a])
        self.visit_counts[s] += 1
        self.cfg.epsilon = max(self.cfg.epsilon_min, self.cfg.epsilon * self.cfg.epsilon_decay)

    def save(self, file_or_path: Union[str, IO[bytes]]):
        def _dump(path):
            states = np.array(list(self.Q1.keys()), dtype=np.int64)
            if len(states)==0:
                q1 = np.zeros((0, self.cfg.n_actions), dtype=np.float32)
                q2 = np.zeros((0, self.cfg.n_actions), dtype=np.float32)
                visits = np.zeros((0,), dtype=np.int64)
            else:
                order = states.argsort()
                states = states[order]
                q1 = np.stack([self.Q1[int(s)] for s in states], axis=0)
                q2 = np.stack([self.Q2[int(s)] for s in states], axis=0)
                visits = np.array([self.visit_counts[int(s)] for s in states], dtype=np.int64)
            np.savez_compressed(path, states=states, Q1=q1, Q2=q2, visits=visits, cfg=np.array([
                self.cfg.n_actions, self.cfg.alpha, self.cfg.gamma, self.cfg.epsilon,
                self.cfg.epsilon_min, self.cfg.epsilon_decay, self.cfg.seed
            ], dtype=np.float64))
        if hasattr(file_or_path, "write"):
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
                _dump(tmp.name); tmp.flush(); tmp.seek(0)
                file_or_path.write(tmp.read()); os.unlink(tmp.name)
        else:
            _dump(file_or_path)

    def load(self, file_or_path: Union[str, IO[bytes]]):
        if hasattr(file_or_path, "read"):
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
                tmp.write(file_or_path.read()); tmp.flush(); path = tmp.name
            data = np.load(path, allow_pickle=True); os.unlink(path)
        else:
            data = np.load(file_or_path, allow_pickle=True)
        states = data["states"]; Q1 = data["Q1"]; Q2 = data["Q2"]; visits = data["visits"]
        self.Q1, self.Q2, self.visit_counts = {}, {}, {}
        for i, s in enumerate(states):
            self.Q1[int(s)] = Q1[i].astype(np.float32, copy=True)
            self.Q2[int(s)] = Q2[i].astype(np.float32, copy=True)
            self.visit_counts[int(s)] = int(visits[i])
        cfg_arr = data["cfg"]
        self.cfg.n_actions = int(cfg_arr[0])
        self.cfg.alpha = float(cfg_arr[1]); self.cfg.gamma = float(cfg_arr[2])
        self.cfg.epsilon = float(cfg_arr[3]); self.cfg.epsilon_min = float(cfg_arr[4])
        self.cfg.epsilon_decay = float(cfg_arr[5]); self.cfg.seed = int(cfg_arr[6])

    def qsum_dense(self):
        if len(self.Q1)==0:
            return np.array([]), np.zeros((0, self.cfg.n_actions), dtype=np.float32)
        states = np.array(sorted(self.Q1.keys()), dtype=np.int64)
        qsum = np.stack([self.Q1[int(s)] + self.Q2[int(s)] for s in states], axis=0)
        return states, qsum
