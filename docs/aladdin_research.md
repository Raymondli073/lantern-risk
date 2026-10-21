# What BlackRock's Aladdin does for risk and systematic investing, from public sources

This note collects what BlackRock and the press describe publicly about Aladdin's risk and
systematic-investing machinery, and maps each piece to its counterpart in Lantern. Facts are
attributed to their sources at the end. Nothing here comes from proprietary documentation.

## 1. What Aladdin is

- **Origin.** Aladdin ("Asset, Liability, Debt and Derivative Investment Network") started
  in 1988 as BlackRock's internal risk system for mortgage and fixed-income portfolios.
  BlackRock began offering it to external clients in 2000 through BlackRock Solutions. Wider
  adoption followed the 2008 crisis. [1]
- **Scale.** Roughly $21.6 trillion of assets across about 30,000 portfolios had been cited
  by 2020. [1] Aladdin Wealth runs risk analytics on **more than 15 million portfolios
  every night**, plus about 3 million API-driven analyses during the day. [6]
- **Components.**
  - Aladdin Risk: the analytics engine for public and private markets.
  - Aladdin Wealth: risk tools for wealth managers.
  - eFront (acquired 2019) for private markets.
  - Aladdin Climate (2020).
  - Preqin (2025) for private-markets data.
  - Aladdin Copilot, a generative-AI interface launched in 2023. [1]
- **Business.** Technology-services revenue reached about $1.1bn in the first half of 2026
  and **$588m in Q3 2026 (+14% year on year)**. BlackRock is integrating AI across Aladdin
  and building a single public plus private "whole-portfolio" view. [9, 10]
- **Technology.** Java/Scala calculation engines, Spark for batch runs, Hadoop/Hive, Solr and
  Cassandra for storage, and Kubernetes on Azure. [1, 6]

## 2. The risk model

- **Factors.** BlackRock's materials describe a multi-factor risk model with **over 2,200
  distinct risk factors** spanning equity, rates, credit, FX and alternatives. [3] Aladdin
  Wealth lets users express views across **3,000+ risk factors**. [2, 4] Factors are chosen
  to be *interpretable, explanatory, consistent across regimes and effective* at improving
  forecasts. [3]
- **Estimation.** The Scenario Tester's statistical model uses **ten years of factor
  history** to estimate factor volatilities and correlations. [3]
- **Exposures.** Exposures combine several sources: observable security terms (yield,
  currency, domicile), fundamentals (financial ratios), technicals (price behavior,
  liquidity), and security-level pricing models. [3]
- **Fund modeling.** Funds are modeled three ways [3]:
  - **holdings**, for equity funds;
  - **building blocks**, for fixed-income and multi-asset funds mapped to representative
    portfolios;
  - **return regression**, for alternative funds.
- **Equity model.** The BlackRock Fundamental Risk for Equities model (**BFRE World**) scores
  value, momentum, quality, size, low volatility and growth as z-scores from −3 to +3. [5]
- **Computation.** Portfolio risk is the quadratic form **E · C · Eᵀ**: an exposure vector
  against a dense ~3,000 × 3,000 covariance matrix (about 9 million entries). The matrix is
  computed once each night after the close and cached, then reused thousands of times per
  portfolio analysis. [6]
- **Methods.** The methods named include ex-ante (tracking-error) risk, historical VaR,
  Monte Carlo VaR, stress testing and factor decomposition. Press coverage describes millions
  of Monte Carlo simulations run for clients. [7]
- **Modeling team.** Aladdin Financial Engineering has about **400 modelers and
  technologists**. It covers pricing and prepayment, risk, return attribution, liquidity,
  optimization and portfolio construction, and scenario analysis and simulation. Its
  Portfolio Risk team builds "linear factor models, VaR methodologies, volatility and
  covariance matrix estimation, and portfolio stress testing and scenario analytics". [8]

## 3. Scenarios

- **Market-Driven Scenarios (MDS)** follow five steps [2, 4]:
  1. Define a market event.
  2. Build a range of outcomes ("good, bad, ugly").
  3. Specify explicit shocks to the most directly affected variables.
  4. Use the risk model to **imply shocks to all remaining factors from their modeled
     relationships**, check plausibility against history, and iterate.
  5. Revalue portfolios.
- **Sizing.** Scenario Tester shocks are sized at about **one standard deviation** of the
  underlying factor, which corresponds to roughly a 16% probability per year. Each result
  carries a **prediction error**, a 95% band that widens when the scenario factors explain
  the portfolio poorly. [3]

## 4. Systematic investing on Aladdin

- **BlackRock Systematic** (formerly Systematic Active Equity) screens more than 15,000
  stocks. It combines fundamental, sentiment, macro and ESG signals with machine learning,
  using a training platform it began building in 2014. [11]
- **Factor Workbench.** The factor-investing team built an **Aladdin Factor Workbench**.
  It translates asset allocations into factor exposures and back, so investors can target
  exposures under constraints such as liquidity. The team also uses Aladdin's daily
  transaction-cost estimates for more than 50,000 securities. [12]
- **Macro factors.** BlackRock's framework for asset owners describes asset classes through
  **economic growth, real rates, inflation, credit, emerging-markets and liquidity**
  factors, alongside style factors. [13]

## 5. Criticism and open problems

- **Systemic risk.** Regulators such as the UK FCA, and critics, warn about concentration
  risk and "groupthink". When many institutions use the same risk signals, they may trade
  in correlated ways during crises. [1]
- **Plausibility.** Hypothetical scenarios depend on judgment about which shocks to run and
  how large to make them, which is why MDS adds plausibility checks against history. [2, 3]

## 6. How Lantern maps to this, and where my research adds to it

| Aladdin element | Lantern | Research contribution |
|---|---|---|
| Thousands of factors, 10-year history | 27 public-data factors, EWMA with fast vol and slow correlation, 10-year lookback, stressed covariance | Multivariate realized-volatility work (MSc) informs the covariance design. |
| Return-regression fund mapping | EWMA **adaptive LASSO** exposure selection with post-selection refit | Thesis project on mutual-fund factor selection with adaptive LASSO (64k funds). |
| E·C·Eᵀ risk, Euler decomposition | `risk/parametric.py` | — |
| Monte Carlo VaR | Gaussian / Student-t / filtered-HS / regime-mixture simulators with full revaluation | Cross-asset factor simulation. |
| MDS: explicit shocks → implied shocks → plausibility → prediction error | `scenarios/propagation.py`, with Mahalanobis plausibility and security-level shocks | — |
| Scenario design by judgment ("good / bad / ugly") | **Explosiveness-triggered bubble-burst scenarios**: the trigger is a right-tailed unit-root test with known size, and the shock size comes from the PSY-dated run-up | PhD: real-time explosiveness detection (PSY BSADF) |
| Systemic-risk / groupthink concern | **Common-bubble monitor**: real-time PCA (CCIPCA) of a cross-asset panel, followed by BSADF | PhD chapter: IPCA-RLS real-time common-bubble detection |
| Factor Workbench | Factor-targeting optimizer on `B S B' + D`, plus risk parity, minimum variance and maximum diversification | — |
| Systematic factor timing | **TSFE** (explosiveness) vs TSFM (momentum) on the factor zoo, net of costs, with NW alphas | PhD paper: "Timing the Factor Zoo: Explosiveness Beyond Momentum" |

## Sources

1. Wikipedia, "Aladdin (BlackRock)": https://en.wikipedia.org/wiki/Aladdin_(BlackRock)
2. BlackRock, "The making of a market-driven scenario": https://www.blackrock.com/aladdin/platforms/solutions/aladdin-wealth/making-of-a-market-driven-scenario
3. BlackRock, "Scenario Analysis Methodology": https://www.blackrock.com/us/financial-professionals/tools/scenarios-methodology
4. BlackRock, Aladdin Risk product page: https://www.blackrock.com/aladdin/products/aladdin-risk
5. BlackRock, "Equity Factor Exposures Methodology" (BFRE World): https://www.blackrock.com/us/financial-professionals/tools/factor-box-methodology
6. InfoQ, "Portfolio Analysis at Scale: Running Risk and Analytics on 15+ Million Portfolios Every Day": https://www.infoq.com/presentations/portfolio-analysis-scale/
7. Financial Advisor, "Aladdin brings BlackRock's big-data-based risk management to advisors": https://www.fa-mag.com/news/aladdin-brings-blackrock-s-big-data-based-risk-management-to-advisors-32160.html
8. BlackRock Careers, Aladdin Financial Engineering portfolio-risk roles: https://careers.blackrock.com/job/new-york/portfolio-risk-modeling-associate/45831/86293977168
9. BlackRock Form 10-Q, Q2 2026: https://www.sec.gov/Archives/edgar/data/0002012383/000119312526337177/blk-20260630.htm
10. BlackRock Q3 2026 earnings call coverage: https://www.webull.com/news/15748319224603648
11. Jio BlackRock, "Data-driven investing explained: inside BlackRock's Systematic Active Equity approach": https://www.jioblackrockamc.com/learn/data-driven-investing-explained-inside-blackrock-systematic-active-equity-approach
12. Risk.net, "Asset manager of the year, systematic investing: BlackRock": https://www.risk.net/awards/5367791/asset-manager-of-the-year-systematic-investing-blackrock
13. BlackRock, "Risk factor investing" (Currents, via PSERS): https://www.psers.pa.gov/About/Investment/Documents/PPMAIRC%202018/5%20Risk%20Factor%20Investing%20Fall%202012%20Currents.pdf
