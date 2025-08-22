
import os, sys
import streamlit as st

# ---------------- Page setup ----------------
st.set_page_config(page_title="Smart Bolsa — Robô de Ações", layout="wide")
st.title("Smart Bolsa — Robô de Ações")
st.caption("Selecione as ações e clique **Executar**. Mostramos um veredito simples e um gráfico.")

# Garantir import do pacote ao rodar via streamlit_app.py
if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importar módulos do projeto com proteção
try:
    import numpy as np
    import pandas as pd
    from rltrader.data import load_builtin_sample, split_walk_forward
    from rltrader.env import MultiAssetEnv
    from rltrader.agent import DoubleQAgent, AgentConfig
except Exception as boot_err:
    st.error("Não consegui iniciar os componentes do app.")
    st.exception(boot_err)
    st.stop()

def brl(x: float) -> str:
    s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

# ---------- Dados embutidos (sem Excel) ----------
@st.cache_data(show_spinner=False)
def load_all_prices():
    df = load_builtin_sample()
    return df

all_prices = load_all_prices()
# Ordena tickers em ordem alfabética
ALL_TICKERS = sorted(list(all_prices.columns))

# ---- Helper: ler seleção ao vivo dos checkboxes ----
def _selected_from_state():
    return [t for t in ALL_TICKERS if st.session_state.get(f"sel_{t}", False)]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Ações disponíveis")
    st.caption("Marque **até 11 ações** que deseja incluir no teste.")

    # Removido "Marcar todas"; mantém só "Limpar seleção"
    if st.button("Limpar seleção"):
        for t in ALL_TICKERS:
            st.session_state[f"sel_{t}"] = False

    # Limite de seleção
    LIMIT = 11
    for t in ALL_TICKERS:
        st.session_state.setdefault(f"sel_{t}", False)
    selected_count = sum(1 for t in ALL_TICKERS if st.session_state.get(f"sel_{t}", False))
    st.caption(f"Selecionadas: **{selected_count}/{LIMIT}**")

    # Checkboxes (desabilita novos checks ao atingir o limite)
    for t in ALL_TICKERS:
        key = f"sel_{t}"
        disabled = (selected_count >= LIMIT) and (st.session_state.get(key) is False)
        st.checkbox(t, key=key, disabled=disabled)
    selected_count = sum(1 for t in ALL_TICKERS if st.session_state.get(f"sel_{t}", False))

    st.header("Configurações")
    initial_cash = st.number_input("Saldo inicial", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")

    # Interpretação em % (usuário digita 0,10 para 0,10%)
    cost_pct_input = st.number_input("Custo % por negociação (ex.: 0,10 para 0,10%)",
                                     min_value=0.0, value=0.10, step=0.01, format="%.4f")
    cost_pct = cost_pct_input / 100.0  # fração para o ambiente (0,10% -> 0.0010)
    st.caption(f"Interpretação: **{cost_pct_input:.4f}%** → fração **{cost_pct:.6f}** por negociação.")
    if cost_pct > 0.02:
        st.warning("Custo % muito alto (>2% por negociação). Isso pode destruir o resultado.", icon="⚠️")

    # Deixa claro que é em reais
    cost_fixed = st.number_input("Custo fixo por ativo (R$)", min_value=0.0, value=0.50, step=0.5, format="%.2f")

    # Intervalo de análise — travado entre 01/01/2020 e 20/08/2025
    min_d = pd.Timestamp("2020-01-01").date()
    max_d = pd.Timestamp("2025-08-20").date()

    cini, cfim = st.columns(2)
    default_start = max(min_d, all_prices.index.min().date())
    start_date = cini.date_input("Início", value=default_start, min_value=min_d, max_value=max_d)
    end_date   = cfim.date_input("Fim",    value=max_d,        min_value=min_d, max_value=max_d)

    if start_date > end_date:
        st.warning("Ajustei o intervalo: início estava depois do fim.")
        start_date, end_date = end_date, start_date

    # guarda no estado para usarmos no treino e no gráfico
    st.session_state["date_range"] = (str(start_date), str(end_date))

# ---------------- Estado ----------------
ss = st.session_state
episodes = 400

# ---------------- Botões principais ----------------
c1, c2 = st.columns([1,1])
run_btn = c1.button("Executar (aprender e avaliar)", type="primary")
reset_btn = c2.button("Limpar resultados")
if reset_btn:
    for k in ("agent","hist_train","hist_test","initial_cash_used"):
        ss.pop(k, None)
    st.success("Resultados limpos.")

# ---------------- Execução protegida ----------------
if run_btn:
    try:
        selected_live = _selected_from_state()
        # Valida limite de 11
        if len(selected_live) == 0:
            st.error("Selecione pelo menos 1 ação."); st.stop()
        if len(selected_live) > 11:
            st.error(f"Você selecionou {len(selected_live)} ações. Selecione **no máximo 11**."); st.stop()

        with st.spinner("Treinando e avaliando (400 rodadas)..."):
            # recorta a janela escolhida
            s0, s1 = ss.get("date_range", (str(all_prices.index.min().date()), str(all_prices.index.max().date())))
            s0, s1 = pd.to_datetime(s0), pd.to_datetime(s1)

            window = all_prices.loc[s0:s1]
            prices = window[selected_live].dropna(how="all")
            if len(prices) < 30:
                st.error(f"Poucos dados ({len(prices)} linhas) para as ações escolhidas."); st.stop()

            # hiperparâmetros internos (ocultos)
            seed = 42
            alpha, gamma = 0.10, 0.99
            epsilon, epsilon_min, epsilon_decay = 0.10, 0.01, 0.999
            risk_penalty, turnover_penalty = 0.001, 0.0

            # walk-forward 70/30
            train, test = split_walk_forward(prices, 0.7)

            env_train = MultiAssetEnv(train, initial_cash=initial_cash, cost_pct=cost_pct,
                                      cost_fixed=cost_fixed, risk_penalty=risk_penalty,
                                      turnover_penalty=turnover_penalty, seed=seed)

            cfg = AgentConfig(n_actions=env_train.n_actions, alpha=alpha, gamma=gamma,
                              epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, seed=seed)
            agent = DoubleQAgent(cfg)

            progress = st.progress(0)
            for ep in range(episodes):
                s = env_train.reset(); done=False
                while not done:
                    a = agent.policy(s)
                    ns, r, done, info = env_train.step(a)
                    agent.update(s, a, r, ns, done); s = ns
                if (ep+1) % 20 == 0:
                    progress.progress(min(1.0, (ep+1)/episodes))

            env_test = MultiAssetEnv(test, initial_cash=initial_cash, cost_pct=cost_pct,
                                     cost_fixed=cost_fixed, risk_penalty=risk_penalty,
                                     turnover_penalty=turnover_penalty, seed=seed)
            s = env_test.reset(); done=False
            while not done:
                a = agent.policy(s); ns, r, done, info = env_test.step(a); s = ns

            ss["agent"] = agent
            ss["hist_train"] = env_train.history
            ss["hist_test"]  = env_test.history
            ss["initial_cash_used"] = float(initial_cash)
        st.success("Pronto! O robô aprendeu e foi avaliado.")
    except Exception as e:
        st.error("Erro durante a execução."); st.exception(e)

# ---------------- Veredito simples + GRÁFICO FULL PERÍODO ----------------
if ss.get("hist_test") is not None:
    import matplotlib.pyplot as plt
    import pandas as pd

    icash = float(ss.get("initial_cash_used") or 0.0)
    nav_end = float(ss["hist_test"]["nav"][-1])
    pct = (100*(nav_end-icash)/icash) if icash else 0.0

    # faixas
    thr_pos = icash * 1.02
    thr_neg = icash * 0.98

    if pct >= 2:
        selo, titulo = "✅", "APROVADO"
    elif pct > -2:
        selo, titulo = "🟡", "NEUTRO"
    else:
        selo, titulo = "❌", "REPROVADO"

    st.subheader(f"{selo} Veredito: {titulo}")

    # ---- Texto explicativo (apenas itens que o usuário pode alterar) ----
    if titulo == "NEUTRO":
        st.markdown(f"""
**Significa:** o robô não conseguiu provar vantagem nem prejuízo claro no teste.  
Quando aparece **🟡 NEUTRO**, a variação do patrimônio no **período de teste** ficou entre **−2% e +2%** — ou seja, terminou praticamente no zero-a-zero, dentro de uma “faixa morta” que usamos para não tirar conclusões com base em ruído.

**Como ler em R$**

Se o saldo inicial foi **{brl(icash)}**, então:

- **Aprovado** seria > **{brl(thr_pos)}**  
- **Reprovado** seria < **{brl(thr_neg)}**  
- **Neutro** é entre **{brl(thr_neg)}** e **{brl(thr_pos)}**.  
*(Fórmula: variação = 100 × (final − inicial) / inicial)*

**O que tentar em seguida (você pode alterar no app):**
- Reduzir **custos**: Custo % por negociação e Custo fixo por ativo (R$).  
- **Rever a quantidade de ações (para cima ou para baixo) e a combinação de ações.**  
- Mudar o **período** (Início/Fim) para testar outra janela de mercado.

**Resumo:** **NEUTRO = inconclusivo**. O modelo não foi bem nem mal; só não encontrou um *edge* confiável naquele teste.
""")
    elif titulo == "APROVADO":
        st.markdown(f"""
**Significa:** o robô ficou **acima da faixa neutra** no período de teste — há um **sinal positivo** naquele recorte (ainda assim, é simulação).

**Como ler em R$**

Se o saldo inicial foi **{brl(icash)}**, então:

- **Aprovado**: > **{brl(thr_pos)}** (seu resultado: **{brl(nav_end)}**)  
- **Neutro**: entre **{brl(thr_neg)}** e **{brl(thr_pos)}**  
- **Reprovado**: < **{brl(thr_neg)}**  
*(Fórmula: variação = 100 × (final − inicial) / inicial)*

**Próximos passos (você pode alterar no app):**
- Manter **custos realistas** (ajuste Custo % e Custo fixo se necessário).  
- **Rever a quantidade de ações (para cima ou para baixo) e a combinação de ações.**  
- Validar com **outras janelas** (Início/Fim) para checar consistência.
""")
    else:  # REPROVADO
        st.markdown(f"""
**Significa:** o robô ficou **abaixo da faixa neutra** — **perdeu dinheiro** no período de teste (simulação).

**Como ler em R$**

Se o saldo inicial foi **{brl(icash)}**, então:

- **Reprovado**: < **{brl(thr_neg)}** (seu resultado: **{brl(nav_end)}**)  
- **Neutro**: entre **{brl(thr_neg)}** e **{brl(thr_pos)}**  
- **Aprovado**: > **{brl(thr_pos)}**  
*(Fórmula: variação = 100 × (final − inicial) / inicial)*

**O que tentar (você pode alterar no app):**
- Reduzir **custos**: Custo % por negociação e Custo fixo por ativo (R$).  
- **Rever a quantidade de ações (para cima ou para baixo) e a combinação de ações.**  
- Mudar o **período** (Início/Fim) para testar outra janela.
""")

    # --- Séries de treino e teste + gráfico cobrindo toda a janela ---
    def make_series(hist):
        d = pd.to_datetime(hist["date"])
        v = pd.Series(hist["nav"], index=d).astype(float)
        return v

    series_train = make_series(ss["hist_train"]) if ss.get("hist_train") is not None else pd.Series(dtype=float)
    series_test  = make_series(ss["hist_test"])  if ss.get("hist_test")  is not None else pd.Series(dtype=float)

    # costura: escala teste para continuar do fim do treino
    if len(series_train) and len(series_test):
        factor = series_train.iloc[-1] / (icash if icash != 0 else 1.0)
        series_test = series_test * factor

    series_all = pd.concat([series_train, series_test]).sort_index()

    s0, s1 = ss.get("date_range")
    s0, s1 = pd.to_datetime(s0), pd.to_datetime(s1)
    series_all = series_all[(series_all.index >= s0) & (series_all.index <= s1)]

    # preenche só o FIM para cobrir a janela inteira (sem "linha reta" no começo)
    if len(series_all) and series_all.index[-1] < s1:
        pad_idx_right = pd.bdate_range(series_all.index[-1] + pd.Timedelta(days=1), s1)
        if len(pad_idx_right):
            series_all = pd.concat([series_all, pd.Series(series_all.iloc[-1], index=pad_idx_right)])

    if series_all.empty:
        st.warning("Sem dados suficientes no intervalo selecionado para exibir o gráfico.")
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(series_all.index, series_all.values, linewidth=1.8)
        # marca fim do treino, se existir
        if len(series_train):
            split_dt = series_train.index[-1]
            ax.axvline(split_dt, linestyle=":", linewidth=1.2)
        # marca início/fim da janela escolhida
        ax.axvline(s0, linestyle="--", linewidth=1.0)
        ax.axvline(s1, linestyle="--", linewidth=1.0)
        ax.set_xlim(s0, s1)
        ax.set_title("Patrimônio — Treino + Teste (janela selecionada)")
        ax.set_xlabel("Data")
        ax.set_ylabel("R$")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ---- Disclaimer destacado ----
st.markdown(
    """
    <div style="margin-top:1rem;padding:12px 16px;border-left:6px solid #d9480f;background:#fff3cd;color:#664d03;border-radius:6px;">
    <b>⚠️ Projeto educacional. Não é recomendação de investimento.</b>
    </div>
    """,
    unsafe_allow_html=True,
)
