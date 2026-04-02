"""
generate_report.py — post-run visualizations and optional Gemini LLM summary.

Called automatically by run_pipeline.py after all gates complete.
Can also be run standalone:
  python generate_report.py --run-dir output/runs/2026-04-03_10-00-00/

Outputs written to <run-dir>/:
  radar.png       — per-model pass-rate radar across all gates
  heatmap.png     — segment × gate PASS/FAIL heatmap (all models)
  llm_report.txt  — Gemini quality summary (requires GEMINI_API_KEY env var)

Dependencies (base env):
  matplotlib, numpy, pandas, google-genai (pip install google-genai)
"""

import os
import sys
import logging
import argparse

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive — safe for subprocess / server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

log = logging.getLogger("pipeline")

# ── Gate config ────────────────────────────────────────────────────────────────
# Which column in each gate's per_segment_results.csv carries the pass/fail value
_PASS_COL: dict[str, str] = {
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

GATE_ORDER: list[str] = list(_PASS_COL.keys())

# Display labels for gate axes (shorter for radar readability)
_GATE_LABEL: dict[str, str] = {
    "wer":             "WER",
    "nisqa":           "NISQA",
    "utmos":           "UTMOS",
    "speaker_sim":     "SPK SIM",
    "ser":             "SER",
    "arousal_valence": "ARO/VAL",
    "pitch":           "PITCH",
    "duration":        "DURATION",
    "vad":             "VAD",
    "amplitude":       "AMP",
    "accent":          "ACCENT",
}

# Status → integer encoding for heatmap
_STATUS_ENC: dict[str, int] = {
    "PASS":      3,
    "NEAR_MISS": 2,
    "FAIL":      1,
    "SKIP":      0,
}

# Status → fill colour (hex)
_STATUS_COLOR: dict[int, str] = {
    3: "#43a047",   # green  — PASS
    2: "#fb8c00",   # orange — NEAR_MISS
    1: "#e53935",   # red    — FAIL
    0: "#9e9e9e",   # grey   — SKIP / ERROR / missing
}

# Short single-character annotation printed inside each cell
_STATUS_ANN: dict[str, str] = {
    "PASS":      "✓",
    "NEAR_MISS": "~",
    "FAIL":      "✗",
    "SKIP":      "—",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _normalise_status(val) -> str:
    """Map any gate pass-column value to one of PASS / NEAR_MISS / FAIL / SKIP."""
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


def _load_gate(run_dir: str, gate_key: str) -> pd.DataFrame | None:
    """
    Load per_segment_results.csv for one gate.
    Returns a DataFrame with columns [Model, Sample, Status] or None if unavailable.
    """
    path = os.path.join(run_dir, gate_key, "per_segment_results.csv")
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.warning("[report] Could not read %s: %s", path, e)
        return None

    pass_col = _PASS_COL.get(gate_key)
    if not pass_col or pass_col not in df.columns:
        log.warning("[report] Pass column '%s' not found in %s — skipping gate", pass_col, path)
        return None

    if "Model" not in df.columns or "Sample" not in df.columns:
        log.warning("[report] Model/Sample columns missing in %s — skipping gate", path)
        return None

    df = df[["Model", "Sample", pass_col]].copy()
    df["Status"] = df[pass_col].apply(_normalise_status)
    return df[["Model", "Sample", "Status"]]


def _build_matrix(run_dir: str):
    """
    Build a per-model status matrix from all available gates.

    Returns:
        matrix  — dict[model_name] → DataFrame(index=sample, columns=gate_key)
        models  — sorted list of model names
        samples — sorted list of sample names
        gates   — list of gate keys that had data, in GATE_ORDER
    """
    gates_available = []
    gate_frames: dict[str, pd.DataFrame] = {}

    for gate in GATE_ORDER:
        df = _load_gate(run_dir, gate)
        if df is not None:
            gates_available.append(gate)
            gate_frames[gate] = df

    if not gates_available:
        return {}, [], [], []

    all_models  = sorted({m for df in gate_frames.values() for m in df["Model"].unique()})
    all_samples = sorted({s for df in gate_frames.values() for s in df["Sample"].unique()})

    matrix: dict[str, pd.DataFrame] = {}
    for model in all_models:
        rows: dict[str, dict[str, str]] = {}
        for sample in all_samples:
            row: dict[str, str] = {}
            for gate in gates_available:
                df = gate_frames[gate]
                match = df[(df["Model"] == model) & (df["Sample"] == sample)]
                row[gate] = match["Status"].iloc[0] if len(match) == 1 else "SKIP"
            rows[sample] = row
        matrix[model] = pd.DataFrame.from_dict(rows, orient="index")  # index=sample, cols=gate

    return matrix, all_models, all_samples, gates_available


# ── Radar chart ────────────────────────────────────────────────────────────────
def _make_radar(run_dir: str, matrix: dict, models: list, gates: list) -> None:
    """
    Polar chart with one axis per gate and one polygon per model.
    Y-axis value = fraction of non-SKIP segments that passed.
    """
    n = len(gates)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#fafafa")

    cmap = matplotlib.colormaps["tab10"].resampled(max(len(models), 1))

    for idx, model in enumerate(models):
        df = matrix[model]
        rates: list[float] = []
        for gate in gates:
            if gate not in df.columns:
                rates.append(0.0)
                continue
            col    = df[gate]
            total  = (col != "SKIP").sum()
            passes = (col == "PASS").sum()
            rates.append(passes / total if total > 0 else 0.0)

        values = rates + rates[:1]
        color  = cmap(idx)
        ax.plot(angles_closed, values, "o-", linewidth=2.2, color=color, label=model)
        ax.fill(angles_closed, values, alpha=0.10, color=color)

    ax.set_xticks(angles)
    ax.set_xticklabels([_GATE_LABEL.get(g, g) for g in gates], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25 %", "50 %", "75 %", "100 %"], fontsize=7.5, color="#555")
    ax.yaxis.set_tick_params(pad=20)
    ax.set_title("Gate Pass Rate by Model", fontsize=15, fontweight="bold", pad=25)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.40, 1.15),
        fontsize=11,
        framealpha=0.85,
    )
    ax.grid(color="#ccc", linewidth=0.8)

    out = os.path.join(run_dir, "radar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("[report] Radar chart  → %s", out)


# ── Heatmap ────────────────────────────────────────────────────────────────────
def _make_heatmap(run_dir: str, matrix: dict, models: list, samples: list, gates: list) -> None:
    """
    Grid chart: rows = (model, sample), columns = gate.
    Cells are coloured by status; short annotation printed inside.
    Models are visually separated by a white divider row.
    """
    # Build row list with optional divider rows between models
    row_data: list[dict | None] = []   # None = divider row
    row_labels: list[str]       = []

    for midx, model in enumerate(models):
        if midx > 0:
            row_data.append(None)
            row_labels.append("")
        df = matrix[model]
        for sample in samples:
            row: dict[str, str] = {}
            for gate in gates:
                if gate not in df.columns or sample not in df.index:
                    row[gate] = "SKIP"
                else:
                    row[gate] = df.loc[sample, gate]
            row_data.append(row)
            row_labels.append(f"{model}  /  {sample}")

    n_rows = len(row_data)
    n_cols = len(gates)

    cell_h = 0.50   # inches per data row
    cell_w = 0.90   # inches per gate column
    fig_h  = max(4.0, n_rows * cell_h + 2.5)
    fig_w  = max(6.0, n_cols * cell_w + 3.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    for r, row in enumerate(row_data):
        if row is None:
            # Divider row — draw a thick horizontal line
            ax.axhline(r, color="white", linewidth=4, zorder=5)
            continue
        for c, gate in enumerate(gates):
            status = row.get(gate, "SKIP")
            enc    = _STATUS_ENC.get(status, 0)
            color  = _STATUS_COLOR[enc]
            ann    = _STATUS_ANN.get(status, "?")

            rect = mpatches.FancyBboxPatch(
                (c - 0.46, r - 0.42), 0.92, 0.84,
                boxstyle="round,pad=0.04",
                linewidth=0,
                facecolor=color,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                c, r, ann,
                ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
                zorder=3,
            )

    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(-0.7, n_rows - 0.3)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [_GATE_LABEL.get(g, g) for g in gates],
        fontsize=9, fontweight="bold",
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_yticks([i for i, r in enumerate(row_data) if r is not None])
    ax.set_yticklabels(
        [lbl for lbl in row_labels if lbl],
        fontsize=8.5,
    )
    ax.invert_yaxis()
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Segment × Gate Pass / Fail Heatmap", fontsize=13, fontweight="bold", pad=35)

    patches = [
        mpatches.Patch(color=_STATUS_COLOR[3], label="PASS (✓)"),
        mpatches.Patch(color=_STATUS_COLOR[2], label="NEAR MISS (~)"),
        mpatches.Patch(color=_STATUS_COLOR[1], label="FAIL (✗)"),
        mpatches.Patch(color=_STATUS_COLOR[0], label="SKIP / ERROR (—)"),
    ]
    ax.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        fontsize=8.5,
        framealpha=0.9,
    )

    out = os.path.join(run_dir, "heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("[report] Heatmap       → %s", out)


# ── LLM summary ────────────────────────────────────────────────────────────────
def _make_llm_report(run_dir: str, matrix: dict, models: list, gates: list) -> None:
    """
    Call Gemini to produce a plain-text quality assessment.
    Silently skips if GEMINI_API_KEY is not set or google-genai is not installed.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.info("[report] GEMINI_API_KEY not set — skipping LLM report")
        return

    try:
        from google import genai as _genai
    except ImportError:
        log.warning("[report] google-genai not installed — run: pip install google-genai")
        return

    # ── Build prompt ───────────────────────────────────────────────────────────
    lines: list[str] = [
        "You are a TTS quality analyst reviewing dubbing quality metrics.",
        "Below is an automated evaluation report for Hindi-to-English TTS dubbing.",
        "",
        f"Run directory : {os.path.basename(run_dir)}",
        f"Models tested : {', '.join(models)}",
        f"Gates run     : {', '.join(gates)}",
        "",
        "=== PASS RATES (PASS count / clean segments, NM = near-miss count) ===",
        "",
    ]

    # Table header
    header = f"{'Gate':<16}" + "".join(f"  {m:<16}" for m in models)
    lines.append(header)
    lines.append("-" * len(header))

    for gate in gates:
        row = f"{gate:<16}"
        for model in models:
            df = matrix.get(model)
            if df is None or gate not in df.columns:
                row += f"  {'N/A':<16}"
                continue
            col    = df[gate]
            total  = (col != "SKIP").sum()
            passes = (col == "PASS").sum()
            nm     = (col == "NEAR_MISS").sum()
            fails  = (col == "FAIL").sum()
            cell   = f"{passes}/{total} ({nm} NM, {fails} F)"
            row   += f"  {cell:<16}"
        lines.append(row)

    lines += ["", "=== PROBLEM SEGMENTS (FAIL and NEAR_MISS only) ===", ""]
    found_problems = False
    for model in models:
        df = matrix.get(model)
        if df is None:
            continue
        model_probs: list[str] = []
        for gate in gates:
            if gate not in df.columns:
                continue
            for sample in df.index:
                status = df.loc[sample, gate]
                if status in ("FAIL", "NEAR_MISS"):
                    model_probs.append(f"  {sample:<25} {gate:<16} → {status}")
        if model_probs:
            found_problems = True
            lines.append(f"Model: {model}")
            lines.extend(model_probs)
            lines.append("")

    if not found_problems:
        lines.append("  (no failures or near-misses detected)")

    lines += [
        "",
        "=== YOUR TASK ===",
        "Based on the data above, provide a concise quality report covering:",
        "1. Overall verdict for each model: production-ready / needs work / reject",
        "2. Which model performs best overall and which gate differentiates them most",
        "3. The top 2–3 failure patterns (recurring gate failures or sample-level issues)",
        "4. Specific, actionable recommendations for the TTS/audio team",
        "Keep the response under 400 words. Be direct and specific — avoid generic advice.",
    ]

    prompt = "\n".join(lines)

    # ── Call Gemini (try models in preference order for free-tier compatibility) ──
    # Try models in preference order — lite variants more likely to have free-tier quota
    _MODELS = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-pro-latest",
    ]
    _RETRY_CODES = ("429", "404", "RESOURCE_EXHAUSTED", "NOT_FOUND")

    client     = _genai.Client(api_key=api_key)
    report_text: str | None = None

    for model_name in _MODELS:
        try:
            response    = client.models.generate_content(model=model_name, contents=prompt)
            report_text = response.text
            log.info("[report] Gemini model used: %s", model_name)
            break
        except Exception as e:
            err_str = str(e)
            if any(code in err_str for code in _RETRY_CODES):
                reason = "quota exceeded" if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str else "not found"
                log.warning("[report] %s %s, trying next model…", model_name, reason)
                continue
            log.error("[report] Gemini call failed on %s: %s", model_name, e)
            return   # unexpected error, don't retry

    if report_text is None:
        log.error("[report] All Gemini models unavailable. Run again later or check your API key.")
        return

    out = os.path.join(run_dir, "llm_report.txt")
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write("=== PROMPT SENT TO GEMINI ===\n\n")
            f.write(prompt)
            f.write("\n\n=== GEMINI RESPONSE ===\n\n")
            f.write(report_text)
    except OSError as e:
        log.error("[report] Could not write LLM report: %s", e)
        return

    log.info("[report] LLM report   → %s", out)


# ── Public entry point ─────────────────────────────────────────────────────────
def generate(run_dir: str) -> None:
    """Generate all report artefacts for a completed pipeline run."""
    log.info("[report] Generating report for run: %s", run_dir)

    matrix, models, samples, gates = _build_matrix(run_dir)

    if not gates:
        log.warning("[report] No gate output found in %s — skipping report", run_dir)
        return

    log.info("[report] %d model(s), %d sample(s), %d gate(s)",
             len(models), len(samples), len(gates))

    _make_radar(run_dir, matrix, models, gates)
    _make_heatmap(run_dir, matrix, models, samples, gates)
    _make_llm_report(run_dir, matrix, models, gates)

    log.info("[report] Report complete.")


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate radar chart, heatmap, and Gemini report for a pipeline run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run output directory (e.g. output/runs/2026-04-03_10-00-00/)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: directory not found: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    generate(args.run_dir)
