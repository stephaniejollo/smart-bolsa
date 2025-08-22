# Smart Bolsa — Robô de Ações

![screenshot png](https://github.com/user-attachments/assets/6c3d5587-56bb-454e-a162-a0715e2d1bf4)
  
**Teste agora:** https://smart-bolsa.streamlit.app/

## O que é
O **Smart Bolsa** é um app simples e didático para experimentar **estratégias de alocação** em ações brasileiras usando **Aprendizado por Reforço (Double Q-Learning)**.  
Você escolhe até **11 ações**, define custos e o período de análise, e o app treina no início da janela e **avalia no final (walk-forward 70/30)**.  
No fim, você recebe um **veredito claro**:

- ✅ **Aprovado**: desempenho do período de teste > +2%  
- 🟡 **Neutro**: entre −2% e +2% (praticamente “zero a zero”)  
- ❌ **Reprovado**: < −2%

Tudo com **gráficos** e **explicações em português** — pensado para quem não é técnico.

---

## Destaques
- **Escolha de papéis** (até 11 entre os 20 disponíveis).
- **Walk-forward 70/30** para evitar “overfit visual”.
- **Custos realistas**: % por negociação + custo fixo por ativo.
- **Alocações discretas** e **rebalanceamento** com custos.
- **Penalização de risco/turnover** leve, para reduzir ganância em mercados voláteis.
- **Período limitável** entre **2020-01-01** e **2025-08-20** (dados embutidos).

> ⚠️ **Projeto educacional. Não é recomendação de investimento.**  
> O objetivo é aprender sobre teste de estratégias de forma segura e transparente.

---

## Como usar online
1. Acesse **https://smart-bolsa.streamlit.app/**  
2. Marque as ações (até 11), ajuste custos e período.  
3. Clique em **Executar (aprender e avaliar)**.  
4. Leia o **veredito** e veja o **gráfico de patrimônio**.

---

## Executar localmente
> Requer **Python 3.11** (recomendado).

```bash
git clone https://github.com/SEU-USUARIO/smart-bolsa.git
cd smart-bolsa
pip install -r requirements.txt
streamlit run streamlit_app.py

Observações

O dataset embutido (rltrader/sample_prices.csv ou src/rltrader/sample_prices.csv) cobre 2020-01-01 a 2025-08-20.

Se o seu pacote estiver em src/, defina PYTHONPATH=src ao rodar em nuvem (ou inclua sys.path.append(...,"src") no topo do streamlit_app.py).

Dados

O app traz um conjunto embutido de preços (20 tickers brasileiros).
Você pode testar diferentes combinações de papéis e custos e comparar o resultado.

Entendendo o veredito

✅ Aprovado: a estratégia ganhou no período de teste (> +2%).

🟡 Neutro: ficou entre −2% e +2% — sem evidência consistente de vantagem.

❌ Reprovado: a estratégia perdeu (< −2%).

Exemplo de leitura:
Saldo inicial R$ 100.000 →
Aprovado se > R$ 102.000 | Reprovado se < R$ 98.000 | Neutro entre R$ 98.000 e R$ 102.000.
Fórmula: variação = 100 × (final − inicial) / inicial.

❔ Dúvidas comuns:

Não entendo a curva só em parte do período.
O gráfico usa somente a janela escolhida e marca a fronteira treino/teste.

Posso mandar meu próprio Excel?
A versão simplificada usa os dados embutidos para evitar fricção.

Contribuições
Sugestões e issues são bem-vindas! Abra um PR ou registre uma issue com a ideia/bug.

Licença
Consulte o arquivo LICENSE.
