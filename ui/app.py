
from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

st.set_page_config(
    page_title="Adaptive Latent Multi-Agent Reasoning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0D4772"
BLUE = "#2A6F9E"
LIGHT = "#F5F8FB"
ORANGE = "#A95900"
GREEN = "#0B7A53"
RED = "#B42318"
TEXT = "#18333F"

st.markdown(
    f"""
    <style>
      .stApp {{ background: #ffffff; color: {TEXT}; }}
      .block-container {{ max-width: 1450px; padding-top: 1.2rem; padding-bottom: 2.5rem; }}
      .hero {{
        background: linear-gradient(100deg, {NAVY}, #123C5A);
        padding: 1.4rem 1.7rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1rem;
      }}
      .hero h1 {{ margin:0; font-size: 2.0rem; font-weight: 700; }}
      .hero p {{ margin:.45rem 0 0 0; opacity:.92; }}
      .card {{
        background: {LIGHT};
        border: 1px solid #DCE6ED;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        min-height: 108px;
      }}
      .metric-title {{ font-size:.82rem; color:#5E7380; text-transform:uppercase; letter-spacing:.04em; }}
      .metric-value {{ font-size:1.75rem; font-weight:700; color:{NAVY}; margin-top:.2rem; }}
      .small-note {{ color:#607681; font-size:.88rem; }}
      .success-box {{ background:#EDF8F3; border-left:4px solid {GREEN}; padding:.8rem 1rem; border-radius:8px; }}
      .warn-box {{ background:#FFF7ED; border-left:4px solid {ORANGE}; padding:.8rem 1rem; border-radius:8px; }}
      .info-box {{ background:#EFF6FF; border-left:4px solid {BLUE}; padding:.8rem 1rem; border-radius:8px; }}
      .agent {{
        border:1px solid #DCE6ED; border-radius:12px; padding:1rem; background:white;
      }}
      .stop {{ color:{RED}; font-weight:700; }}
      .continue {{ color:{GREEN}; font-weight:700; }}
      div[data-testid="stMetricValue"] {{ color:{NAVY}; }}
      footer {{visibility:hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    records = pd.read_csv(DATA / "phase9_recovered_console_records.csv")
    summary = pd.read_csv(DATA / "phase9_final_performance_summary.csv")
    controller = pd.read_csv(DATA / "phase9_controller_decisions.csv")
    paired = pd.read_csv(DATA / "phase9_paired_statistical_comparison.csv")
    return records, summary, controller, paired

records, summary, controller, paired = load_data()

st.sidebar.markdown("## Adaptive Latent MAS")
st.sidebar.caption("M.Tech 3rd Semester Research Interface")
page = st.sidebar.radio(
    "Navigate",
    [
        "Research Dashboard",
        "Benchmark Explorer",
        "Controller Analysis",
        "Statistical Evidence",
        "Architecture",
        "Reproducibility",
        "Kaggle Connection",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Frozen Phase-9 system")
st.sidebar.code("Qwen/Qwen3-4B\nthreshold = 0.0\ncontroller = gradient_boosting_huber", language="text")

def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def metric_card(title, value, note=""):
    st.markdown(
        f'<div class="card"><div class="metric-title">{title}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="small-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

def correctness_symbol(value):
    return "✓" if bool(value) else "✗"

if page == "Research Dashboard":
    hero(
        "Adaptive Latent Multi-Agent Reasoning",
        "Planner → Critic → Solver | Real Phase-9 unseen-test evidence | 512 benchmark problems",
    )

    comb = summary[summary.dataset.eq("combined")].set_index("system")
    stat = paired[
        (paired.dataset.eq("combined")) &
        (paired.comparison.eq("adaptive_vs_fixed_20"))
    ].iloc[0]

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Adaptive Accuracy", f"{comb.loc['adaptive','accuracy']*100:.2f}%", "320 / 512 correct")
    with c2: metric_card("Saving vs Fixed-20", f"{comb.loc['adaptive','latent_step_reduction_vs_fixed20_percent']:.2f}%", "latent reasoning steps")
    with c3: metric_card("Accuracy gain vs Fixed-20", f"+{stat.accuracy_difference*100:.2f} pp", "paired unseen test")
    with c4: metric_card("Exact McNemar p", f"{stat.mcnemar_exact_p:.6f}", "statistically significant vs Fixed-20")

    st.markdown("### Final Phase-9 Comparison")
    plot = summary[summary.dataset.eq("combined")].copy()
    plot["Accuracy (%)"] = 100 * plot["accuracy"]
    plot["System"] = plot["system"].map({"fixed_10":"Fixed-10","fixed_20":"Fixed-20","adaptive":"Adaptive"})
    fig = px.bar(plot, x="System", y="Accuracy (%)", text="Accuracy (%)",
                 title="Combined Accuracy — 512 Unseen Test Problems")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis_range=[0,75], height=390, margin=dict(t=55,b=30))
    st.plotly_chart(fig, use_container_width=True)

    left,right = st.columns(2)
    with left:
        st.markdown("#### Accuracy by Dataset")
        ds = summary[~summary.dataset.eq("combined")].copy()
        ds["Accuracy (%)"] = 100 * ds.accuracy
        ds["System"] = ds.system.map({"fixed_10":"Fixed-10","fixed_20":"Fixed-20","adaptive":"Adaptive"})
        ds["Dataset"] = ds.dataset.map({"gsm8k":"GSM8K","arc_challenge":"ARC-Challenge"})
        fig2 = px.bar(ds, x="Dataset", y="Accuracy (%)", color="System", barmode="group")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)
    with right:
        st.markdown("#### Mean Latent Computation")
        steps = plot[["System","mean_latent_steps"]].copy()
        fig3 = px.bar(steps, x="System", y="mean_latent_steps", text="mean_latent_steps",
                      labels={"mean_latent_steps":"Mean latent steps"})
        fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig3.update_layout(yaxis_range=[0,68], height=380)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<div class="success-box"><b>Key completed finding.</b> Relative to Fixed-20, '
        'the learned adaptive controller improves combined accuracy by 6.25 percentage points '
        'while reducing mean latent computation by 18.26%. The exact paired McNemar test gives '
        '<b>p = 0.004455</b>.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="warn-box"><b>Important scientific qualification.</b> Adaptive does not '
        'significantly outperform Fixed-10 overall. Fixed-10 is also cheaper. The key observation '
        'is that the preferred reasoning depth is dataset-dependent, motivating the next progressive '
        '5→10→15→20 controller.</div>',
        unsafe_allow_html=True,
    )

elif page == "Benchmark Explorer":
    hero(
        "Benchmark Explorer",
        "Inspect real per-question Phase-9 outputs recovered from the completed Kaggle run.",
    )
    d1,d2,d3 = st.columns([1.1,1.2,2.2])
    with d1:
        ds_filter = st.selectbox("Dataset", ["All","GSM8K","ARC-Challenge"])
    filtered = records.copy()
    if ds_filter == "GSM8K":
        filtered = filtered[filtered.dataset.eq("gsm8k")]
    elif ds_filter == "ARC-Challenge":
        filtered = filtered[filtered.dataset.eq("arc_challenge")]
    with d2:
        correctness = st.selectbox("Adaptive result", ["All","Correct","Incorrect"])
    if correctness == "Correct":
        filtered = filtered[filtered.adaptive_correct]
    elif correctness == "Incorrect":
        filtered = filtered[~filtered.adaptive_correct]
    with d3:
        search = st.text_input("Search example ID or question preview")
    if search:
        mask = (
            filtered.example_id.astype(str).str.contains(search, case=False, na=False)
            | filtered.question_preview.astype(str).str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    options = filtered["problem_number"].tolist()
    if not options:
        st.warning("No matching benchmark records.")
        st.stop()
    selected = st.selectbox(
        "Problem",
        options,
        format_func=lambda x: f"[{x:03d}/512] "
                              f"{records.loc[records.problem_number.eq(x),'dataset'].iloc[0].upper()} | "
                              f"{records.loc[records.problem_number.eq(x),'example_id'].iloc[0]}",
    )
    row = records[records.problem_number.eq(selected)].iloc[0]

    st.markdown("### Question")
    st.info(row.question_preview)

    st.markdown("### Three-System Comparison")
    cols = st.columns(3)
    systems = [
        ("Fixed-10","fixed_10"),
        ("Fixed-20","fixed_20"),
        ("Adaptive","adaptive"),
    ]
    for col,(label,key) in zip(cols,systems):
        with col:
            ok = bool(row[f"{key}_correct"])
            st.markdown(f"#### {label} {'✓' if ok else '✗'}")
            st.metric("Prediction", str(row[f"{key}_prediction"]))
            st.metric("Latent steps", int(row[f"{key}_steps"]))
            st.caption(
                f"P={int(row[f'{key}_planner_steps'])} · "
                f"C={int(row[f'{key}_critic_steps'])} · "
                f"S={int(row[f'{key}_solver_steps'])}"
            )
            a,b = st.columns(2)
            a.metric("Output tokens", int(row[f"{key}_out_tokens"]))
            b.metric("Latency", f"{row[f'{key}_latency_s']:.2f}s")
            st.caption(f"Peak GPU: {row[f'{key}_gpu_gb']:.2f} GB")

    st.markdown("### Adaptive Controller Decisions")
    acols = st.columns(3)
    for col, role in zip(acols, ["planner","critic","solver"]):
        decision = row[f"{role}_decision"]
        utility = row[f"{role}_utility"]
        steps = int(row[f"adaptive_{role}_steps"])
        with col:
            klass = "continue" if decision == "CONTINUE" else "stop"
            st.markdown(
                f'<div class="agent"><div class="metric-title">{role.upper()}</div>'
                f'<div class="{klass}" style="font-size:1.25rem">{decision}</div>'
                f'<div style="font-size:1.35rem;font-weight:700;margin-top:.35rem">{steps} steps</div>'
                f'<div class="small-note">predicted marginal utility = {utility:+.4f}</div></div>',
                unsafe_allow_html=True,
            )
    st.caption(
        f"Compute tier: {row.compute_tier} · "
        f"Saving vs Fixed-20: {row.saving_vs_fixed20_pct:.2f}% · "
        f"Controller overhead: {row.controller_ms:.2f} ms"
    )
    st.warning(
        "The console archive stored a shortened question preview rather than the complete benchmark text. "
        "The UI deliberately does not invent missing text or reference answers."
    )

elif page == "Controller Analysis":
    hero(
        "Controller Analysis",
        "How the learned controller allocates computation independently to Planner, Critic, and Solver.",
    )
    view = controller.copy()
    view["Continue rate (%)"] = 100 * view.continue_rate
    view["Dataset"] = view.dataset.map({"gsm8k":"GSM8K","arc_challenge":"ARC-Challenge"})
    view["Agent"] = view.agent_role.str.title()
    fig = px.bar(
        view, x="Agent", y="Continue rate (%)", color="Dataset", barmode="group",
        title="CONTINUE Rate at the 10-Step Decision Checkpoint",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        util = records.melt(
            id_vars=["dataset"],
            value_vars=["planner_utility","critic_utility","solver_utility"],
            var_name="Agent", value_name="Predicted marginal utility"
        )
        util["Agent"] = util["Agent"].str.replace("_utility","",regex=False).str.title()
        fig2 = px.box(util, x="Agent", y="Predicted marginal utility", points=False,
                      title="Predicted Utility Distribution")
        fig2.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        tier = records.groupby(["dataset","compute_tier"]).size().reset_index(name="n")
        tier["Dataset"] = tier.dataset.map({"gsm8k":"GSM8K","arc_challenge":"ARC-Challenge"})
        fig3 = px.bar(tier, x="compute_tier", y="n", color="Dataset", barmode="group",
                      category_orders={"compute_tier":["LOW","MEDIUM","HIGH"]},
                      title="Adaptive Compute-Tier Distribution")
        st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        controller.assign(
            continue_rate_pct=lambda x: 100*x.continue_rate,
            stop_rate_pct=lambda x: 100*x.stop_rate,
        )[["dataset","agent_role","n","continue_rate_pct","stop_rate_pct","mean_selected_steps","mean_predicted_utility"]],
        use_container_width=True,
        hide_index=True,
    )

elif page == "Statistical Evidence":
    hero(
        "Statistical Evidence",
        "Paired evaluation of Adaptive against the two fixed computation policies.",
    )
    st.dataframe(paired, use_container_width=True, hide_index=True)

    main = paired[(paired.dataset=="combined") & (paired.comparison=="adaptive_vs_fixed_20")].iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy difference", f"{main.accuracy_difference*100:+.2f} pp")
    c2.metric("95% bootstrap CI", f"[{main.accuracy_difference_ci95_low*100:+.2f}, {main.accuracy_difference_ci95_high*100:+.2f}]")
    c3.metric("Exact McNemar p", f"{main.mcnemar_exact_p:.6f}")
    c4.metric("Mean steps saved", f"{main.mean_latent_steps_saved:.2f}")

    st.markdown(
        '<div class="info-box"><b>Interpretation:</b> The Adaptive-vs-Fixed-20 accuracy difference '
        'is positive, its 95% bootstrap interval excludes zero, and the exact McNemar test is significant. '
        'This supports a paired improvement over Fixed-20 on the locked 512-problem evaluation.</div>',
        unsafe_allow_html=True,
    )

elif page == "Architecture":
    hero(
        "System Architecture",
        "Homogeneous Qwen3-4B agents with role-specific adaptive latent computation and KV working-memory transfer.",
    )
    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap;margin:1.6rem 0;">
          <div class="agent" style="width:170px;text-align:center"><b>Question</b></div>
          <div style="font-size:26px">→</div>
          <div class="agent" style="width:190px;text-align:center"><b>Planner</b><br>latent reasoning</div>
          <div style="font-size:26px">→</div>
          <div class="agent" style="width:190px;text-align:center"><b>Controller</b><br>STOP / CONTINUE</div>
          <div style="font-size:26px">→</div>
          <div class="agent" style="width:190px;text-align:center"><b>Critic</b><br>latent reasoning</div>
          <div style="font-size:26px">→</div>
          <div class="agent" style="width:190px;text-align:center"><b>Controller</b></div>
          <div style="font-size:26px">→</div>
          <div class="agent" style="width:190px;text-align:center"><b>Solver</b><br>latent reasoning</div>
          <div style="font-size:26px">→</div>
          <div class="agent" style="width:170px;text-align:center"><b>Final Answer</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a,b = st.columns(2)
    with a:
        st.markdown("### Latent computation")
        st.latex(r"h_t=f(\tilde e_t,KV_{t-1})")
        st.latex(r"\tilde e_{t+1}=W_a h_t")
        st.write(
            "The final-layer hidden state is realigned and fed back as the next latent input. "
            "Intermediate reasoning is not decoded into natural-language messages."
        )
    with b:
        st.markdown("### Learned halting")
        st.latex(r"\Delta U_{10}=\mathrm{NLL}_{10}-\mathrm{NLL}_{20}")
        st.latex(r"\hat{\Delta U}>0\Rightarrow CONTINUE,\qquad \hat{\Delta U}\le0\Rightarrow STOP")
        st.write(
            "The same learned Gradient Boosting Huber controller is applied independently to "
            "Planner, Critic, and Solver using temporal latent-state features plus agent role."
        )

    st.markdown("### Current vs next research stage")
    c1,c2 = st.columns(2)
    with c1:
        st.info("**Completed Adaptive-2Level**\n\nEach agent: 10 → STOP or CONTINUE → 20.")
    with c2:
        st.warning("**Pre-declared next stage**\n\nProgressive learned halting: 5 → 10 → 15 → 20. Not yet claimed as completed.")

elif page == "Reproducibility":
    hero(
        "Research Integrity & Reproducibility",
        "The interface separates completed evidence from planned work and exposes the frozen evaluation protocol.",
    )
    checks = [
        ("Backbone", "Qwen/Qwen3-4B"),
        ("Pinned revision", "1cfa9a7208912126459214e8b04321603b3df60c"),
        ("Final controller", "gradient_boosting_huber"),
        ("Decision threshold", "0.0"),
        ("Phase-9 test size", "512 = 256 GSM8K + 256 ARC-Challenge"),
        ("Controller retrained on test", "NO"),
        ("Threshold tuned on test", "NO"),
        ("Reference used online", "NO"),
        ("True 10→20 continuation", "VERIFIED"),
        ("Regression tests", "72 / 72 PASS"),
        ("Sealed remainder", "1,063 GSM8K + 916 ARC-Challenge"),
    ]
    st.dataframe(pd.DataFrame(checks, columns=["Protocol item","Value"]), hide_index=True, use_container_width=True)
    st.success("The UI data in Results Mode were recovered from the completed Phase-9 Kaggle notebook output; no benchmark values are fabricated.")
    st.caption("Source evidence: notebooks/phase9_completed_reference.ipynb")

elif page == "Kaggle Connection":
    hero(
        "Kaggle Connection",
        "Use Kaggle as the remote research-compute and artifact source while the Streamlit interface runs locally.",
    )
    st.markdown("### Recommended architecture")
    st.code(
        """Local Windows PC
  ├─ Streamlit UI
  ├─ Git repository
  └─ Kaggle CLI
        │
        ├─ authenticate securely
        ├─ pull/push notebooks or scripts
        ├─ start Kaggle GPU kernel jobs
        └─ download generated outputs
""",
        language="text",
    )

    if st.button("Run local Kaggle connection check"):
        cmd = [sys.executable, str(ROOT / "scripts/kaggle_check.py")]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            st.success("Kaggle CLI authentication check passed.")
        else:
            st.error("Kaggle CLI check failed.")
        st.code((p.stdout + "\n" + p.stderr).strip(), language="text")

    st.markdown("### Security rule")
    st.warning(
        "Never commit kaggle.json, API tokens, Hugging Face tokens, .env secrets, model weights, "
        "or private Kaggle credentials to GitHub. The repository .gitignore is configured for this."
    )
    st.markdown("### Remote live inference")
    st.info(
        "Results Mode is ready now and is the recommended professor demo because it is immediate and reproducible. "
        "The repository also contains the Kaggle remote-job scaffold. Live Qwen3-4B inference should be run as a "
        "Kaggle GPU job rather than pretending the local laptop is the inference server."
    )
