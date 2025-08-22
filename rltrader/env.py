
import numpy as np
import pandas as pd

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def volatility_regime(df: pd.DataFrame, window: int = 20) -> pd.Series:
    ret = df.pct_change()
    idx = ret.mean(axis=1).rolling(window).std()
    q1, q2 = idx.quantile(0.33), idx.quantile(0.66)
    regime = pd.Series(0, index=idx.index)
    regime[idx > q1] = 1
    regime[idx > q2] = 2
    return regime.fillna(0).astype(int)

def build_state_index(prices: pd.DataFrame) -> pd.Series:
    fast, slow = 10, 30
    feats = []
    for col in prices.columns:
        r = rsi(prices[col])
        r_bin = pd.cut(r, bins=[-1, 30, 70, 101], labels=[0,1,2]).astype(int)
        ma_fast = prices[col].rolling(fast).mean()
        ma_slow = prices[col].rolling(slow).mean()
        cross = (ma_fast > ma_slow).astype(int)
        feats.append(r_bin.astype(int).rename(f"rsi_{col}"))
        feats.append(cross.astype(int).rename(f"cross_{col}"))
    reg = volatility_regime(prices).rename("reg")
    X = pd.concat(feats + [reg], axis=1).dropna()
    base_sizes = []
    for col in prices.columns:
        base_sizes += [3,2]
    base_sizes += [3]
    multipliers = [1]
    for b in base_sizes[:-1]:
        multipliers.append(multipliers[-1]*b)
    multipliers = np.array(multipliers, dtype=int)
    ordered = []
    for col in prices.columns:
        ordered += [f"rsi_{col}", f"cross_{col}"]
    ordered += ["reg"]
    arr = X[ordered].to_numpy(dtype=int)
    code = (arr * multipliers[:arr.shape[1]]).sum(axis=1)
    code = pd.Series(code, index=X.index, name="state")
    return code

def enumerate_actions(n: int = 3):
    actions = []
    zeros = tuple(0.0 for _ in range(n))
    actions.append(zeros)
    for i in range(n):
        w = [0.0]*n; w[i] = 1.0; actions.append(tuple(w))
    for i in range(n):
        w = [0.0]*n; w[i] = 0.5; actions.append(tuple(w))
    for i in range(n):
        for j in range(i+1, n):
            w = [0.0]*n; w[i] = w[j] = 0.5; actions.append(tuple(w))
    return actions

def turnover(prev_w, new_w) -> float:
    prev = np.array(prev_w, dtype=float)
    new = np.array(new_w, dtype=float)
    return float(np.abs(prev - new).sum())

class MultiAssetEnv:
    def __init__(self, prices: pd.DataFrame, initial_cash=100_000.0, cost_pct=0.001, cost_fixed=0.0,
                 risk_penalty=0.001, turnover_penalty=0.0, seed=42):
        assert isinstance(prices, pd.DataFrame) and len(prices) > 10, "Dados insuficientes"
        self.prices = prices.copy()
        self.assets = list(prices.columns)
        self.n_assets = len(self.assets)
        self.actions = enumerate_actions(self.n_assets)
        self.n_actions = len(self.actions)
        self.initial_cash = float(initial_cash)
        self.cost_pct = float(cost_pct) / 100.0 if cost_pct > 0.5 else float(cost_pct)
        self.cost_fixed = float(cost_fixed)
        self.risk_penalty = float(risk_penalty)
        self.turnover_penalty = float(turnover_penalty)
        self.rng = np.random.default_rng(seed)

        self.returns = self.prices.pct_change().fillna(0.0)
        self.state_idx = build_state_index(self.prices)
        self.returns = self.returns.loc[self.state_idx.index]
        self.reset()

    def reset(self):
        if self.state_idx is None or len(self.state_idx) == 0:
            raise ValueError("Dados insuficientes para iniciar o episódio (sem estados calculados).")
        self.t = 0
        self.nav = self.initial_cash
        self.weights = tuple([0.0]*self.n_assets)
        self.history = {"date": [], "nav": [], "weights": [], "state": []}
        self.history["date"].append(self.state_idx.index[self.t])
        self.history["nav"].append(self.nav)
        self.history["weights"].append(self.weights)
        self.history["state"].append(int(self.state_idx.iloc[self.t]))
        return int(self.state_idx.iloc[self.t])

    def step(self, action_id: int):
        assert 0 <= action_id < self.n_actions
        target_w = self.actions[action_id]
        move = turnover(self.weights, target_w)
        assets_changed = sum(1 for i in range(self.n_assets) if abs(self.weights[i]-target_w[i])>1e-12)
        traded_value = self.nav * move
        costs = traded_value * self.cost_pct + assets_changed * self.cost_fixed
        self.nav -= costs
        self.weights = target_w
        if self.t+1 >= len(self.returns):
            done = True; reward = 0.0
        else:
            r = self.returns.iloc[self.t+1]
            port_ret = float(np.dot(self.weights, r.values))
            risk_term = self.risk_penalty * float(r.var())
            turnover_term = self.turnover_penalty * move
            reward = self.nav * port_ret - (risk_term*self.nav) - (turnover_term*self.nav)
            self.nav *= (1.0 + port_ret)
            self.t += 1
            done = (self.t >= len(self.state_idx)-1)
        self.history["date"].append(self.state_idx.index[self.t if self.t < len(self.state_idx) else -1])
        self.history["nav"].append(self.nav)
        self.history["weights"].append(self.weights)
        self.history["state"].append(int(self.state_idx.iloc[self.t if self.t < len(self.state_idx) else -1]))
        return int(self.state_idx.iloc[self.t]), float(reward), bool(done), {}

    @property
    def n_states(self):
        # (6 estados por ativo)*3 (regime)
        return (6 ** len(self.assets)) * 3
