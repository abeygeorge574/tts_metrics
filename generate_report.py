"""
generate_report.py — post-run visualizations and optional LLM summary.

Called automatically by run_pipeline.py after all gates complete.
Can also be run standalone:
  python generate_report.py --run-dir output/runs/2026-04-03_10-00-00/

Outputs (written to <run-dir>/):
  radar.png       — per-model pass rate radar across all gates
  heatmap.png     — segment × gate pass/fail heatmap per model
  llm_report.txt  — Gemini summary (only if GEMINI_API_KEY is set)
"""

import os
import sys
import argparse

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Gate config ────────────────────────────────────────────────────────────────
# Maps gate_key → column name in per_segment_results.csv that holds pass/fail
_PASS_COL = {
    "wer":             "Both_Pass",
    "nisqa":           "Final",
    "utmos":           "Pass",
    "speaker_sim":     "Pass",
    "ser":             "Pass",
    "arousal_valence": "Pass",
    "pitch":           "Final Pass",
    "duration":        "Final Pass",
    "vad":             "Final Pass",
    "amplitude":       "Final Pass",
    "accent":          "Final Pass",
}

GATE_ORDER = list(_PASS_COL.keys())

# Numeric encoding for heatmap cells
_ENC = {"PASS": 3, "NEAR_MISS": 2, "FAIL": 1, "SKIP": 0}
_COLORS = {
    3: "#4caf50",   # green  — PASS
    2: "#ff9800",   # orange — NEAR_MISS
    1: "#f44336",   # red    — FAIL
    0: "#9e9e9e",   # grey   — SKIP / ERROR / missing
}
_LABELS = {3: "PASS", 2: "NEAR MISS", 1: "FAIL", 0: "SKIP/ERROR"}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _normalise_pass(val):
    """Convert any gate's pass-column value to PASS / NEAR_MISS / FAIL / SKIP."""
    if pd.isna(val):
        return "SKIP"
    s = str(val).strip().upper()
    if s in ("PASS", "TRUE", "1"):
        return "PASS"
    if s == "NEAR_MISS":
        return "NEAR_MISS"
    if s in ("FAIL", "FALSE", "0"):
        return "FAIL"
    return "SKIP"


def _load_gate(run_dir, gate_key):
    """
    Load per_segment_results.csv for one gate.
    Returns a DataFrame with columns [Model, Sample, Status] or None if missing.
    """
    path = os.path.join(run_dir, gate_key, "per_segment_results.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    pass_col = _PASS_COL.get(gate_key)
    if pass_col is None or pass_col not in df.columns:
        return None

    df["Status"] = df[pass_col].apply(_normalise_pass)
    return df[["Model", "Sample", "Status"]].copy()


def _build_matrix(run_dir):
    """
    Returns:
      matrix  — dict[model] → DataFrame(index=sample, columns=gate)  values=status string
      models  — list of model names
      samples — list of sample names
      gates   — list of gate keys that had data
    """
    gates_with_data = []
    gate_frames = {}

    for gate in GATE_ORDER:
        df = _load_gate(run_dir, gate)
        if df is not None:
            gates_with_data.append(gate)
            gate_frames[gate] = df

    if not gates_with_data:
        return None, [], [], []

    # Collect all models and samples
    all_models  = sorted({row for df in gate_frames.values() for row in df["Model"].unique()})
    all_samples = sorted({row for df in gate_frames.values() for row in df["Sample"].unique()})

    matrix = {}
    for model in all_models:
        rows = {}
        for sample in all_samples:
            row = {}
            for gate in gates_with_data:
                df = gate_frames[gate]
                match = df[(df["Model"] == model) & (df["Sample"] == sample)]
                row[gate] = match["Status"].iloc[0] if len(match) == 1 else "SKIP"
            rows[sample] = row
        matrix[model] = pd.DataFrame(rows).T   # index=sample, columns=gate

    return matrix, all_models, all_samples, gates_with_data


# ── Radar chart ────────────────────────────────────────────────────────────────
def make_radar(run_dir, matrix, models, gates):
    """One polygon per model. Axis = gate. Value = PASS rate (0–1)."""
    if not matrix or not models or not gates:
        return

    n_gates = len(gates)
    angles  = np.linspace(0, 2 * np.pi, n_gates, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    cmap   = plt.cm.get_cmap("tab10", len(models))
    legend = []

    for idx, model in enumerate(models):
        df    = matrix[model]
        rates = []
        for gate in gates:
            if gate not in df.columns:
                rates.append(0.0)
                continue
            statuses = df[gate]
            total    = len(statuses[statuses != "SKIP"])
            passes   = (statuses == "PASS").sum()
            rates.append(passes / total if total > 0 else 0.0)

        rates += rates[:1]   # close the polygon
        color  = cmap(idx)
        ax.plot(angles, rates, "o-", linewidth=2, color=color, label=model)
        ax.fill(angles, rates, alpha=0.12, color=color)
        legend.append(model)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [g.upper().replace("_", "\n") for g in gates],
        size=9
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7, color="grey")
    ax.yaxis.set_tick_params(pad=18)
    ax.set_title("Gate Pass Rate by Model", size=14, pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

    out = os.path.join(run_dir, "radar.png")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] Radar chart → {out}")


# ── Heatmap ────────────────────────────────────────────────────────────────────
def make_heatmap(run_dir, matrix, models, samples, gates):
    """One heatmap combining all models. Rows = model+sample, columns = gate."""
    if not matrix or not models or not gates:
        return

    n_rows = len(models) * len(samples)
    n_cols = len(gates)

    row_labels = []
    data_enc   = []
    data_str   = []

    for model in models:
        df = matrix[model]
        for sample in samples:
            row_labels.append(f"{model} / {sample}")
            enc_row = []
            str_row = []
            for gate in gates:
                if gate not in df.columns or sample not in df.index:
                    enc_row.append(0)
                    str_row.append("—")
                else:
                    status = df.loc[sample, gate]
                    enc_row.append(_ENC.get(status, 0))
                    str_row.append(status[:1] if status != "NEAR_MISS" else "N")
            data_enc.append(enc_row)
            data_str.append(str_row)

    arr = np.array(data_enc, dtype=float)

    fig_h = max(4, n_rows * 0.45 + 2)
    fig_w = max(6, n_cols * 0.9 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw coloured cells
    for r in range(n_rows):
        for c in range(n_cols):
            val   = int(arr[r, c])
            color = _COLORS.get(val, _COLORS[0])
            rect  = mpatches.FancyBboxPatch(
                (c - 0.48, r - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                linewidth=0, facecolor=color, zorder=2
            )
            ax.add_patch(rect)
            label = data_str[r][c]
            ax.text(c, r, label, ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=3)

    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [g.upper().replace("_", "\n") for g in gates],
        fontsize=8, rotation=0
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.invert_yaxis()
    ax.set_title("Segment × Gate Pass/Fail Heatmap", size=13, pad=30, fontweight="bold")

    # Separator lines between models
    for i, model in enumerate(models[:-1]):
        y = (i + 1) * len(samples) - 0.5
        ax.axhline(y, color="white", linewidth=2.5, zorder=4)

    # Legend
    patches = [
        mpatches.Patch(color=_COLORS[3], label="PASS"),
        mpatches.Patch(color=_COLORS[2], label="NEAR MISS (N)"),
        mpatches.Patch(color=_COLORS[1], label="FAIL"),
        mpatches.Patch(color=_COLORS[0], label="SKIP / ERROR"),
    ]
    ax.legend(handles=patches, loc="lower right",
              bbox_to_anchor=(1.0, -0.12), ncol=4, fontsize=8)

    ax.set_aspect("equal")
    out = os.path.join(run_dir, "heatmap.png")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] Heatmap → {out}")


# ── LLM summary ────────────────────────────────────────────────────────────────
def make_llm_report(run_dir, matrix, models, gates):
    """Call Gemini to produce a plain-text quality report. Skips if no API key."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[report] GEMINI_API_KEY not set — skipping LLM report")
        return

    try:
        import google.generativeai as genai
    except ImportError:
        print("[report] google-generativeai not installed — skipping LLM report")
        return

    genai.configure(api_key=api_key)

    # Build structured summary
    lines = [
        "TTS/STS Audio Quality Evaluation",
        f"Run dir: {run_dir}",
        f"Models:  {', '.join(models)}",
        f"Gates:   {', '.join(gates)}",
        "",
        "=== PASS RATES (PASS / total clean segments) ===",
    ]

    # Header row
    lines.append(f"{'Gate':<16} " + "  ".join(f"{m:<10}" for m in models))
    lines.append("-" * (16 + 12 * len(models)))

    for gate in gates:
        row = f"{gate:<16}"
        for model in models:
            df = matrix.get(model)
            if df is None or gate not in df.columns:
                row += f"  {'N/A':<10}"
                continue
            statuses = df[gate]
            total  = len(statuses[statuses != "SKIP"])
            passes = (statuses == "PASS").sum()
            nm     = (statuses == "NEAR_MISS").sum()
            pct    = f"{passes}/{total}" if total > 0 else "—"
            row += f"  {pct:<6} ({nm} NM)"
        lines.append(row)

    lines += ["", "=== FAILED AND NEAR-MISS SEGMENTS ==="]
    for model in models:
        df = matrix.get(model)
        if df is None:
            continue
        problem_rows = []
        for gate in gates:
            if gate not in df.columns:
                continue
            for sample in df.index:
                status = df.loc[sample, gate]
                if status in ("FAIL", "NEAR_MISS"):
                    problem_rows.append(f"  {model} / {sample} / {gate}: {status}")
        if problem_rows:
            lines.append(f"\nModel: {model}")
            lines.extend(problem_rows)

    lines += [
        "",
        "=== TASK FOR LLM ===",
        "You are a TTS quality analyst reviewing dubbing quality metrics.",
        "Based on the data above, provide:",
        "1. An overall quality verdict per model (production-ready / needs work / reject)",
        "2. Which model performs best and why",
        "3. The main failure patterns (which gates fail most, which samples are problematic)",
        "4. Concrete recommendations for the TTS team",
        "Keep the response concise and actionable.",
    ]

    prompt = "\n".join(lines)

    try:
        gemini = genai.GenerativeModel("gemini-1.5-pro")
        response = gemini.generate_content(prompt)
        report_text = response.text
    except Exception as e:
        print(f"[report] Gemini call failed: {e}")
        return

    out = os.path.join(run_dir, "llm_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("=== INPUT TO LLM ===\n\n")
        f.write(prompt)
        f.write("\n\n=== LLM RESPONSE ===\n\n")
        f.write(report_text)

    print(f"[report] LLM report → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def generate(run_dir):
    print(f"[report] Generating report for: {run_dir}")
    matrix, models, samples, gates = _build_matrix(run_dir)

    if not gates:
        print("[report] No gate data found — skipping visualizations")
        return

    print(f"[report] Found {len(models)} model(s), {len(samples)} sample(s), {len(gates)} gate(s)")

    make_radar(run_dir, matrix, models, gates)
    make_heatmap(run_dir, matrix, models, samples, gates)
    make_llm_report(run_dir, matrix, models, gates)
    print("[report] Done.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Generate report for a pipeline run")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run output directory (e.g. output/runs/2026-04-03_10-00-00/)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: run-dir not found: {args.run_dir}")
        sys.exit(1)

    generate(args.run_dir)
