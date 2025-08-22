
import numpy as np

def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        raise RuntimeError("Matplotlib não disponível.") from e

def heatmap_q(qsum: np.ndarray, out_path: str, vmax=None):
    plt = _plt()
    plt.figure(figsize=(10,6))
    if qsum.size == 0:
        plt.text(0.5, 0.5, "Q-table vazia", ha="center", va="center")
    else:
        plt.imshow(qsum, aspect='auto', interpolation='nearest', vmax=vmax)
        plt.colorbar(label="Q1+Q2"); plt.xlabel("Ação"); plt.ylabel("Estado")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_nav(dates, nav, out_path: str):
    plt = _plt(); plt.figure(figsize=(10,5))
    plt.plot(dates, nav); plt.title("Curva de Patrimônio (NAV)")
    plt.xlabel("Data"); plt.ylabel("NAV"); plt.tight_layout(); plt.savefig(out_path); plt.close()
