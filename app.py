"""
Kyrgyzstan Inflation & Prediction Dashboard
Run with: streamlit run inflation_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kyrgyzstan Inflation Analysis",
    page_icon=" ",
    layout="wide",
)

# ── Theme colours ──────────────────────────────────────────────────────────────
C_RED    = "#E24B4A"
C_BLUE   = "#378ADD"
C_GREEN  = "#639922"
C_AMBER  = "#EF9F27"
C_GRAY   = "#888780"
C_PURPLE = "#7F77DD"

# ── Historical data ────────────────────────────────────────────────────────────
HIST = pd.DataFrame({
    "Year": list(range(1996, 2025)),
    "Inflation": [31.2,25.5,10.5,35.9,18.7,6.9,2.1,3.1,4.1,4.3,5.6,10.2,
                  24.5,6.8,7.8,16.6,2.8,6.6,7.5,6.5,0.4,3.2,1.5,1.1,
                  6.3,11.9,13.9,10.8,8.5],
    "GDP_growth": [-5.4,9.9,2.1,3.7,5.4,5.3,0.0,7.0,7.0,-0.2,3.1,8.5,
                   8.4,2.9,-0.5,6.0,-0.1,10.9,4.0,3.9,4.3,4.7,3.8,4.6,
                   -8.4,3.6,7.0,6.3,5.2],
})
HIST["Real_growth"] = ((1 + HIST["GDP_growth"]/100) / (1 + HIST["Inflation"]/100) - 1) * 100
HIST["Inf_exceeds"] = HIST["Inflation"] > HIST["GDP_growth"]

# Purchasing power (base 2000 = $100)
df2k = HIST[HIST["Year"] >= 2000].reset_index(drop=True)
pp = [100.0]
for i in range(1, len(df2k)):
    pp.append(pp[-1] / (1 + df2k.loc[i, "Inflation"] / 100))
df2k["PurchasingPower"] = pp

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Forecast settings")
    st.markdown("---")

    scenario = st.selectbox(
        "Quick scenario",
        ["Custom", "Optimistic (3% inf / 7% GDP)", "Base case (7% inf / 4.5% GDP)", "Pessimistic (13% inf / 2% GDP)"],
        index=2,
    )
    presets = {
        "Optimistic (3% inf / 7% GDP)":  (3.0, 7.0),
        "Base case (7% inf / 4.5% GDP)": (7.0, 4.5),
        "Pessimistic (13% inf / 2% GDP)":(13.0, 2.0),
        "Custom": None,
    }
    if presets[scenario]:
        def_inf, def_gdp = presets[scenario]
    else:
        def_inf, def_gdp = 7.0, 4.5

    inf_rate = st.slider("Inflation assumption (%)", 0.5, 30.0, def_inf, 0.5)
    gdp_rate = st.slider("GDP growth assumption (%)", 0.0, 15.0, def_gdp, 0.5)
    fc_end   = st.slider("Forecast to year", 2026, 2040, 2035, 1)

    st.markdown("---")
    st.caption("Data: World Bank, Kyrgyz Republic 1996–2024")

# ── Derived forecast ───────────────────────────────────────────────────────────
fc_years = list(range(2025, fc_end + 1))
real_fc  = ((1 + gdp_rate/100) / (1 + inf_rate/100) - 1) * 100

last_pp = df2k["PurchasingPower"].iloc[-1]
fc_pp = [last_pp]
for _ in fc_years:
    fc_pp.append(fc_pp[-1] / (1 + inf_rate / 100))
fc_pp = fc_pp[1:]

# ── ML models ─────────────────────────────────────────────────────────────────
X = HIST[["Year"]].values
y = HIST["Inflation"].values

lr = LinearRegression().fit(X, y)
poly = PolynomialFeatures(degree=3)
pr = LinearRegression().fit(poly.fit_transform(X), y)

fc_X  = np.array(fc_years).reshape(-1, 1)
lr_fc = lr.predict(fc_X)
pr_fc = np.clip(pr.predict(poly.transform(fc_X)), 0, 40)

lr_r2   = r2_score(y, lr.predict(X))
pr_r2   = r2_score(y, pr.predict(poly.transform(X)))
lr_rmse = np.sqrt(mean_squared_error(y, lr.predict(X)))
pr_rmse = np.sqrt(mean_squared_error(y, pr.predict(poly.transform(X))))

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 Kyrgyzstan — Inflation & Economic Growth Analysis")
st.markdown("*Inflation, consumer prices (CPI) vs. GDP growth, 1996–2024 · World Bank data*")
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
avg_inf  = HIST["Inflation"].mean()
avg_gdp  = HIST["GDP_growth"].mean()
exceed_n = int(HIST["Inf_exceeds"].sum())
real_color = "normal" if real_fc >= 0 else "inverse"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg inflation 1996–2024",  f"{avg_inf:.1f}%")
k2.metric("Avg GDP growth 1996–2024", f"{avg_gdp:.1f}%")
k3.metric("Years inflation > GDP",    f"{exceed_n} / {len(HIST)}", f"{exceed_n/len(HIST)*100:.0f}%")
k4.metric("Forecast real growth",     f"{real_fc:.2f}%",
          delta=f"inf {inf_rate}% / gdp {gdp_rate}%",
          delta_color=real_color)
k5.metric(f"Purchasing power by {fc_end}", f"${fc_pp[-1]:.1f}",
          delta=f"from $100 in 2000", delta_color="inverse")

st.markdown("---")

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Historical overview",
    "Prediction models",
    "Purchasing power",
    "Forecast table",
])

# ── TAB 1: Historical overview ─────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Inflation & GDP growth over time")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=HIST["Year"], y=HIST["GDP_growth"],
            name="GDP growth", marker_color=[C_GREEN if v >= 0 else C_RED for v in HIST["GDP_growth"]],
            opacity=0.55,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=HIST["Year"], y=HIST["Inflation"],
            name="Inflation (CPI)", line=dict(color=C_RED, width=2.5, dash="dot"),
            mode="lines+markers", marker=dict(size=5),
        ), secondary_y=True)
        fig.update_layout(height=380, margin=dict(t=20, b=20), legend=dict(orientation="h", y=-0.15))
        fig.update_yaxes(title_text="GDP growth (%)", secondary_y=False)
        fig.update_yaxes(title_text="Inflation (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Real inflation-adjusted growth")
        colors = [C_GREEN if v >= 0 else C_RED for v in HIST["Real_growth"]]
        fig2 = go.Figure(go.Bar(
            x=HIST["Year"], y=HIST["Real_growth"],
            marker_color=colors, opacity=0.8,
        ))
        fig2.add_hline(y=0, line_color=C_GRAY, line_width=1)
        fig2.update_layout(height=380, margin=dict(t=20,b=20),
                           yaxis_title="Real growth (inflation-adj. %)")
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Inflation gap (Inflation − GDP growth)")
        gap = HIST["Inflation"] - HIST["GDP_growth"]
        fig3 = go.Figure(go.Bar(
            x=HIST["Year"], y=gap,
            marker_color=[C_RED if v > 0 else C_GREEN for v in gap], opacity=0.8,
        ))
        fig3.add_hline(y=0, line_color=C_GRAY)
        fig3.update_layout(height=300, margin=dict(t=20,b=20), yaxis_title="pp")
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.subheader("Inflation vs. GDP growth — scatter")
        fig4 = px.scatter(
            HIST, x="GDP_growth", y="Inflation", text="Year",
            color="Inf_exceeds",
            color_discrete_map={True: C_RED, False: C_GREEN},
            labels={"GDP_growth":"GDP growth (%)","Inflation":"Inflation (%)","Inf_exceeds":"Inf > GDP"},
        )
        fig4.update_traces(textposition="top center", textfont_size=9, marker_size=8)
        fig4.add_hline(y=0, line_color=C_GRAY, line_width=0.8)
        fig4.add_vline(x=0, line_color=C_GRAY, line_width=0.8)
        # add diagonal
        mn = min(HIST["GDP_growth"].min(), HIST["Inflation"].min()) - 2
        mx = max(HIST["GDP_growth"].max(), HIST["Inflation"].max()) + 2
        fig4.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",
                                   line=dict(color=C_GRAY, dash="dash", width=1),
                                   name="inf = GDP line", showlegend=False))
        fig4.update_layout(height=300, margin=dict(t=20,b=20))
        st.plotly_chart(fig4, use_container_width=True)

# ── TAB 2: Prediction models ───────────────────────────────────────────────────
with tab2:
    st.subheader("Inflation prediction — regression models")

    m1, m2 = st.columns(2)

    with m1:
        st.markdown("**Linear regression**")
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(x=HIST["Year"], y=HIST["Inflation"],
                                     mode="markers", name="Observed",
                                     marker=dict(color=C_GRAY, size=7)))
        fig_lr.add_trace(go.Scatter(x=HIST["Year"], y=lr.predict(X),
                                     mode="lines", name="Linear fit",
                                     line=dict(color=C_BLUE, width=2)))
        fig_lr.add_trace(go.Scatter(x=fc_years, y=lr_fc,
                                     mode="lines+markers", name="Forecast",
                                     line=dict(color=C_BLUE, dash="dash", width=2),
                                     marker=dict(size=5)))
        fig_lr.add_vrect(x0=2024.5, x1=fc_end+0.5, fillcolor=C_BLUE,
                          opacity=0.04, line_width=0)
        fig_lr.add_vline(x=2024.5, line_color=C_GRAY, line_dash="dot")
        fig_lr.update_layout(height=340, margin=dict(t=10,b=10),
                              yaxis_title="Inflation (%)", legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig_lr, use_container_width=True)
        st.caption(f"R² = {lr_r2:.3f} · RMSE = {lr_rmse:.2f} pp")

    with m2:
        st.markdown("**Polynomial regression (degree 3)**")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=HIST["Year"], y=HIST["Inflation"],
                                     mode="markers", name="Observed",
                                     marker=dict(color=C_GRAY, size=7)))
        fig_pr.add_trace(go.Scatter(x=HIST["Year"], y=pr.predict(poly.transform(X)),
                                     mode="lines", name="Polynomial fit",
                                     line=dict(color=C_GREEN, width=2)))
        fig_pr.add_trace(go.Scatter(x=fc_years, y=pr_fc,
                                     mode="lines+markers", name="Forecast",
                                     line=dict(color=C_GREEN, dash="dash", width=2),
                                     marker=dict(size=5)))
        fig_pr.add_vrect(x0=2024.5, x1=fc_end+0.5, fillcolor=C_GREEN,
                          opacity=0.04, line_width=0)
        fig_pr.add_vline(x=2024.5, line_color=C_GRAY, line_dash="dot")
        fig_pr.update_layout(height=340, margin=dict(t=10,b=10),
                              yaxis_title="Inflation (%)", legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig_pr, use_container_width=True)
        st.caption(f"R² = {pr_r2:.3f} · RMSE = {pr_rmse:.2f} pp")

    st.markdown("---")
    st.subheader("Scenario forecast — inflation & GDP growth")

    fig_scen = go.Figure()
    fig_scen.add_trace(go.Scatter(x=HIST["Year"], y=HIST["Inflation"],
                                   mode="lines+markers", name="Inflation (hist)",
                                   line=dict(color=C_RED, width=2), marker=dict(size=4)))
    fig_scen.add_trace(go.Scatter(x=HIST["Year"], y=HIST["GDP_growth"],
                                   mode="lines+markers", name="GDP growth (hist)",
                                   line=dict(color=C_BLUE, width=2, dash="dot"), marker=dict(size=4)))

    scenarios_plot = {
        "Optimistic": (3.0, 7.0, C_GREEN),
        "Base case":  (7.0, 4.5, C_BLUE),
        "Pessimistic":(13.0,2.0, C_RED),
    }
    for name, (si, sg, col) in scenarios_plot.items():
        fig_scen.add_trace(go.Scatter(
            x=[2024]+fc_years, y=[HIST["Inflation"].iloc[-1]]+[si]*len(fc_years),
            mode="lines", name=f"{name} inf={si}%",
            line=dict(color=col, dash="dash", width=2.5),
        ))
        fig_scen.add_trace(go.Scatter(
            x=[2024]+fc_years, y=[HIST["GDP_growth"].iloc[-1]]+[sg]*len(fc_years),
            mode="lines", name=f"{name} GDP={sg}%",
            line=dict(color=col, dash="longdash", width=1.8),
        ))
    # Highlight selected custom
    fig_scen.add_trace(go.Scatter(
        x=[2024]+fc_years, y=[HIST["Inflation"].iloc[-1]]+[inf_rate]*len(fc_years),
        mode="lines+markers", name=f"Selected: inf={inf_rate}%",
        line=dict(color=C_AMBER, width=3), marker=dict(size=6, symbol="diamond"),
    ))
    fig_scen.add_vline(x=2024.5, line_color=C_GRAY, line_dash="dot")
    fig_scen.update_layout(height=400, margin=dict(t=10,b=10),
                            yaxis_title="% annual",
                            legend=dict(orientation="h", y=-0.3, font_size=10))
    st.plotly_chart(fig_scen, use_container_width=True)

    st.subheader("Model evaluation")
    eval_df = pd.DataFrame({
        "Model": ["Linear Regression", "Polynomial (deg 3)"],
        "R²":    [round(lr_r2, 3), round(pr_r2, 3)],
        "RMSE (pp)": [round(lr_rmse, 2), round(pr_rmse, 2)],
        "MAE (pp)":  [round(np.mean(np.abs(y - lr.predict(X))), 2),
                      round(np.mean(np.abs(y - pr.predict(poly.transform(X)))), 2)],
    })
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

# ── TAB 3: Purchasing power ────────────────────────────────────────────────────
with tab3:
    st.subheader("Cumulative purchasing power of $100 (base: 2000)")

    fig_pp = go.Figure()
    fig_pp.add_trace(go.Scatter(
        x=df2k["Year"], y=df2k["PurchasingPower"],
        mode="lines+markers", name="Historical",
        line=dict(color=C_GRAY, width=2.5), marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(136,135,128,0.08)",
    ))
    fig_pp.add_trace(go.Scatter(
        x=[2024]+fc_years, y=[last_pp]+fc_pp,
        mode="lines+markers", name=f"Forecast (inf={inf_rate}%)",
        line=dict(color=C_AMBER, dash="dash", width=2.5), marker=dict(size=6, symbol="diamond"),
        fill="tozeroy", fillcolor="rgba(239,159,39,0.08)",
    ))
    fig_pp.add_vline(x=2024.5, line_color=C_GRAY, line_dash="dot",
                      annotation_text="Forecast →", annotation_position="top right")
    fig_pp.add_hline(y=last_pp, line_color=C_GRAY, line_dash="dot", line_width=0.8)
    fig_pp.update_layout(height=400, margin=dict(t=10,b=10),
                          yaxis_title="Purchasing power (USD)",
                          legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig_pp, use_container_width=True)

    st.subheader("Three-scenario comparison")
    fig_sc2 = go.Figure()
    fig_sc2.add_trace(go.Scatter(x=df2k["Year"], y=df2k["PurchasingPower"],
                                  mode="lines", name="Historical",
                                  line=dict(color=C_GRAY, width=2)))
    sc_cols = {"Optimistic (3%)": (3.0, C_GREEN), "Base (7%)":(7.0, C_BLUE), "Pessimistic (13%)":(13.0, C_RED)}
    for name, (si, col) in sc_cols.items():
        pv = [last_pp]
        for _ in fc_years:
            pv.append(pv[-1]/(1+si/100))
        pv = pv[1:]
        fig_sc2.add_trace(go.Scatter(
            x=[2024]+fc_years, y=[last_pp]+pv,
            mode="lines+markers", name=f"{name} → ${pv[-1]:.1f}",
            line=dict(color=col, dash="dash", width=2.5), marker=dict(size=5),
        ))
    fig_sc2.add_vline(x=2024.5, line_color=C_GRAY, line_dash="dot")
    fig_sc2.update_layout(height=360, margin=dict(t=10,b=10),
                           yaxis_title="Purchasing power ($)",
                           legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_sc2, use_container_width=True)

# ── TAB 4: Forecast table ──────────────────────────────────────────────────────
with tab4:
    st.subheader(f"Year-by-year forecast · {inf_rate}% inflation / {gdp_rate}% GDP growth")

    rows = []
    pp_val = last_pp
    for yr in fc_years:
        pp_val = pp_val / (1 + inf_rate / 100)
        rg = ((1 + gdp_rate/100) / (1 + inf_rate/100) - 1) * 100
        rows.append({
            "Year": yr,
            "Inflation (%)": inf_rate,
            "GDP growth (%)": gdp_rate,
            "Real growth (%)": round(rg, 2),
            "Purch. power of $100": round(pp_val, 2),
            "Inflation > GDP": "Yes ⚠️" if inf_rate > gdp_rate else "No ✅",
        })
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("All-scenario summary")
    all_scen = []
    for sname, (si, sg) in [("Optimistic",(3,7)),("Base case",(7,4.5)),("Pessimistic",(13,2))]:
        pv = last_pp
        for _ in fc_years:
            pv /= (1 + si/100)
        rg = ((1+sg/100)/(1+si/100)-1)*100
        all_scen.append({
            "Scenario": sname,
            "Avg inflation": f"{si}%",
            "Avg GDP growth": f"{sg}%",
            "Real growth / yr": f"{rg:.2f}%",
            f"Purch. power by {fc_end}": f"${pv:.2f}",
        })
    st.dataframe(pd.DataFrame(all_scen), use_container_width=True, hide_index=True)

# ── Team section ──────────────────────────────────────────────────────────────
st.markdown("---")

st.markdown(
    """
    <div style="
        background-color:#1e293b;
        padding:20px;
        border-radius:15px;
        text-align:center;
        margin-top:20px;
        margin-bottom:20px;
        box-shadow:0 0 15px rgba(0,0,0,0.2);
    ">
        <h2 style="color:#EF9F27;">
            Project Team
        </h2>

        <p style="font-size:20px; color:white;">
            👩‍💻 Aidana &nbsp;&nbsp; | &nbsp;&nbsp;
            👩‍💻 Meerim &nbsp;&nbsp; | &nbsp;&nbsp;
            👩‍💻 Aizhan
        </p>

        <p style="color:#94a3b8;">
            Kyrgyzstan Inflation & Economic Forecast Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.caption("Kyrgyzstan Inflation Analysis · World Bank Data · 1996–2024 · Prediction model built with scikit-learn")
