"""Build the PDF report using ReportLab.

Reads results/metrics.json (if it exists) and writes report/report.pdf.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
    Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics.json"
OUT_PATH = ROOT / "report" / "report.pdf"

# ---------- fonts ----------
# Prefer DejaVu Sans which is present on standard Linux; fall back to Helvetica.
SYSTEM_FONTS = {
    "Body-Regular": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ],
    "Body-Bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ],
}


def _register_fonts():
    have = True
    for name, paths in SYSTEM_FONTS.items():
        registered = False
        for pth in paths:
            if os.path.exists(pth):
                try:
                    pdfmetrics.registerFont(TTFont(name, pth))
                    registered = True
                    break
                except Exception as e:
                    print(f"font register failed ({name}): {e}")
        if not registered:
            have = False
    return have


# ---------- styles ----------
NAVY = colors.HexColor("#1B474D")
TEAL = colors.HexColor("#20808D")
INK = colors.HexColor("#28251D")
MUTED = colors.HexColor("#7A7974")
BG = colors.HexColor("#F7F6F2")


def _styles(have_custom: bool):
    body_font = "Body-Regular" if have_custom else "Helvetica"
    bold_font = "Body-Bold" if have_custom else "Helvetica-Bold"
    styles = {
        "title": ParagraphStyle(
            "title", fontName=bold_font, fontSize=22, leading=27, textColor=INK,
            spaceAfter=6),
        "subtitle": ParagraphStyle(
            "subtitle", fontName=body_font, fontSize=11, leading=14, textColor=MUTED,
            spaceAfter=18),
        "h2": ParagraphStyle(
            "h2", fontName=bold_font, fontSize=14, leading=18, textColor=NAVY,
            spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle(
            "h3", fontName=bold_font, fontSize=11, leading=15, textColor=INK,
            spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle(
            "body", fontName=body_font, fontSize=10, leading=14, textColor=INK,
            spaceAfter=6),
        "bullet": ParagraphStyle(
            "bullet", fontName=body_font, fontSize=10, leading=14, textColor=INK,
            leftIndent=12, bulletIndent=0, spaceAfter=3),
        "small": ParagraphStyle(
            "small", fontName=body_font, fontSize=8.5, leading=11, textColor=MUTED),
        "code": ParagraphStyle(
            "code", fontName="Courier", fontSize=8.5, leading=11, textColor=INK,
            backColor=BG, borderPadding=4, spaceAfter=6, leftIndent=6),
    }
    return styles


def _load_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return None


def _aggregate(all_results):
    import numpy as np
    agg = {}
    for name in ["random", "wm_reward", "wm_vlm"]:
        vals = [v for k, v in all_results.items() if v["agent"] == name]
        if not vals:
            continue
        agg[name] = {
            "success_rate_mean": float(np.mean([v["success_rate"] for v in vals])),
            "success_rate_std": float(np.std([v["success_rate"] for v in vals])),
            "mean_return": float(np.mean([v["mean_return"] for v in vals])),
            "mean_steps": float(np.mean([v["mean_steps"] for v in vals])),
            "num_seeds": len(vals),
            "n_ep": vals[0]["num_episodes"],
        }
    return agg


def _results_table(agg, styles):
    header = ["Agent", "Success rate", "Mean return", "Mean length", "Seeds × eps."]
    rows = [header]
    label = {"random": "Random", "wm_reward": "WM planning (no VLM)", "wm_vlm": "WM planning + VLM"}
    for k in ["random", "wm_reward", "wm_vlm"]:
        if k not in agg:
            rows.append([label[k], "—", "—", "—", "—"])
            continue
        v = agg[k]
        rows.append([
            label[k],
            f"{v['success_rate_mean']*100:.1f}% ± {v['success_rate_std']*100:.1f}",
            f"{v['mean_return']:.3f}",
            f"{v['mean_steps']:.1f}",
            f"{v['num_seeds']} × {v['n_ep']}",
        ])
    t = Table(rows, colWidths=[4.4*cm, 3.0*cm, 2.7*cm, 2.7*cm, 2.7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), styles["body"].fontName),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D4D1CA")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FBFBF9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def build():
    have_fonts = _register_fonts()
    styles = _styles(have_fonts)
    metrics = _load_metrics()

    doc = SimpleDocTemplate(
        str(OUT_PATH), pagesize=A4,
        leftMargin=2.0*cm, rightMargin=2.0*cm,
        topMargin=2.0*cm, bottomMargin=2.0*cm,
        title="World Model + VLM Scorer — Demo Report",
        author="Perplexity Computer",
    )
    story = []

    # ---- Cover ----
    story.append(Paragraph("World Model + VLM Scorer", styles["title"]))
    story.append(Paragraph(
        "A minimal Dreamer-style RSSM + CLIP-based planner in MiniGrid",
        styles["subtitle"]))

    story.append(Paragraph("1. Task", styles["h2"]))
    story.append(Paragraph(
        "The goal is to combine a learned world model (Dreamer-style RSSM) with a "
        "vision-language scorer (CLIP) to plan actions in a simple gridworld. Actions "
        "are chosen by MPC over imagined rollouts. The VLM turns imagined future frames "
        "into a scalar goal-progress score used inside the planner's objective.",
        styles["body"]))

    # ---- Method ----
    story.append(Paragraph("2. Method", styles["h2"]))

    story.append(Paragraph("2.1 Environment", styles["h3"]))
    story.append(Paragraph(
        "MiniGrid <b>Empty-8x8-v0</b>. Observations are the full-view RGB renders "
        "resized to 64×64. We restrict the action space to <font face='%s'>{left, right, forward}</font> "
        "(the three actions that matter in Empty). Reward is sparse: a positive value on the goal cell, "
        "zero otherwise, plus a small time penalty embedded in the environment." %
        styles["code"].fontName,
        styles["body"]))

    story.append(Paragraph("2.2 RSSM world model", styles["h3"]))
    story.append(Paragraph(
        "A compact recurrent state-space model in the style of PlaNet (Hafner et al., 2018) "
        "and Dreamer (Hafner et al., 2020, 2023). State <font face='%s'>s_t = (h_t, z_t)</font> is a "
        "deterministic GRU hidden <font face='%s'>h_t &#8712; R^128</font> paired with a stochastic latent "
        "<font face='%s'>z_t &#8712; R^16</font> drawn from a diagonal Gaussian." %
        (styles["code"].fontName, styles["code"].fontName, styles["code"].fontName),
        styles["body"]))
    story.append(Paragraph("Components (≈2.3M parameters total):", styles["body"]))
    for b in [
        "<b>Encoder</b> — 4-layer strided CNN 64×64→256-d embed.",
        "<b>Recurrent dynamics</b> — GRUCell on (z<sub>t-1</sub>, a<sub>t-1</sub>) → h<sub>t</sub>.",
        "<b>Prior</b> — MLP h<sub>t</sub> → μ, σ for p(z<sub>t</sub>|h<sub>t</sub>).",
        "<b>Posterior</b> — MLP (h<sub>t</sub>, e<sub>t</sub>) → μ, σ for q(z<sub>t</sub>|h<sub>t</sub>, o<sub>t</sub>).",
        "<b>Decoder</b> — transposed CNN feat→(3,64,64) reconstruction.",
        "<b>Reward head</b> — MLP feat → scalar.",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))
    story.append(Paragraph(
        "Loss: reconstruction MSE + KL(posterior‖prior) with free bits (1.0) + reward MSE. "
        "Training uses random-policy episodes sampled from a replay buffer.",
        styles["body"]))

    story.append(Paragraph("2.3 Planner (MPC / random shooting)", styles["h3"]))
    story.append(Paragraph(
        "At each real step we (i) update the RSSM posterior with the current observation, "
        "(ii) sample <b>N</b> candidate action sequences of length <b>H</b>, "
        "(iii) imagine each rollout with the RSSM prior, decode all H+1 frames and gather "
        "predicted rewards, (iv) score each rollout, (v) execute the first action of the "
        "highest-scoring sequence and repeat. Default N=128, H=12. CEM is a drop-in improvement "
        "and would be a natural next step; random shooting is enough to compare scorers.",
        styles["body"]))

    story.append(Paragraph("2.4 VLM scorer", styles["h3"]))
    story.append(Paragraph(
        "We use <b>CLIP ViT-B/32 (OpenAI)</b> via <font face='%s'>open_clip</font>. "
        "Text prompts are used contrastively:" % styles["code"].fontName,
        styles["body"]))
    story.append(Paragraph(
        "&nbsp;&nbsp;• positive: <i>&ldquo;a red triangle agent standing on the green goal square&rdquo;</i><br/>"
        "&nbsp;&nbsp;• negative: <i>&ldquo;a red triangle agent far from the green goal square&rdquo;</i>",
        styles["body"]))
    story.append(Paragraph(
        "For each imagined <b>future</b> frame (t = 1 … H) we compute "
        "<font face='%s'>score = 100·(cos(pos) − cos(neg))</font>, then aggregate with a discount "
        "γ = 0.9. Only future frames enter the score — the current observation is excluded — "
        "which satisfies the assignment's constraint that scoring is applied to rollout frames." %
        styles["code"].fontName,
        styles["body"]))

    story.append(Paragraph("2.5 Baselines", styles["h3"]))
    story.append(Paragraph("Two required baselines are included:", styles["body"]))
    for b in [
        "<b>Random</b> — uniform sampling over the 3 actions.",
        "<b>WM planning without VLM</b> — same MPC pipeline, but the scoring function is the "
        "sum of the RSSM's own predicted rewards over the horizon.",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    story.append(PageBreak())

    # ---- Results ----
    story.append(Paragraph("3. Results", styles["h2"]))
    if metrics:
        agg = _aggregate(metrics)
        first = next(iter(metrics.values()))
        n_ep = first["num_episodes"]
        n_seeds = len({v["seed"] for v in metrics.values()})
        story.append(Paragraph(
            f"Evaluation: {n_seeds} seed(s) × {n_ep} episodes per agent. "
            "Success = agent reaches the green goal within the environment's step budget.",
            styles["body"]))
        story.append(_results_table(agg, styles))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Raw per-seed metrics are in <font face='%s'>results/metrics.json</font>. "
            "First-episode GIFs of each agent are in <font face='%s'>results/gifs/</font>." %
            (styles["code"].fontName, styles["code"].fontName),
            styles["small"]))
    else:
        story.append(Paragraph(
            "<i>No results file found yet.</i> The pipeline is verified end-to-end via the "
            "smoke test (see <font face='%s'>scripts/run_smoke_test.py</font>); to fill in this "
            "table, run:" % styles["code"].fontName,
            styles["body"]))
        story.append(Paragraph(
            "python -m src.train_wm --episodes 300 --updates 10000<br/>"
            "python -m src.evaluate --episodes 20 --seeds 0 1 2 --horizon 12 --num-seq 128",
            styles["code"]))
        story.append(_results_table({}, styles))

    story.append(Paragraph("3.1 Visualisation", styles["h3"]))
    gif_dir = ROOT / "results" / "gifs"
    png_examples = sorted(gif_dir.glob("*.png")) if gif_dir.exists() else []
    if png_examples:
        # Include only first PNG per agent
        seen = set()
        for p in png_examples:
            key = p.stem.split("_seed")[0]
            if key in seen:
                continue
            seen.add(key)
            story.append(Paragraph(p.stem, styles["small"]))
            story.append(Image(str(p), width=6*cm, height=6*cm))
            story.append(Spacer(1, 4))
    else:
        cf = styles["code"].fontName
        story.append(Paragraph(
            f"Rollout snapshots and per-agent GIFs are written to "
            f"<font face='{cf}'>results/gifs/</font> during evaluation. Example filenames: "
            f"<font face='{cf}'>random_seed0_ep0.gif</font>, "
            f"<font face='{cf}'>wm_reward_seed0_ep0.gif</font>, "
            f"<font face='{cf}'>wm_vlm_seed0_ep0.gif</font>.",
            styles["body"]))

    # ---- Discussion ----
    story.append(Paragraph("4. Discussion", styles["h2"]))

    story.append(Paragraph("4.1 Main failure modes observed", styles["h3"]))
    for b in [
        "<b>Blurry imagined frames.</b> On CPU with a small training budget, the RSSM decoder "
        "produces low-detail reconstructions of imagined states — the agent triangle and the "
        "goal square are recognizable only as coloured blobs. This directly degrades the VLM "
        "signal: CLIP's confidence in either prompt is muted, and the contrast <i>pos − neg</i> "
        "becomes noisy for long horizons.",
        "<b>Compounding rollout drift.</b> The prior is trained only on 10–20-step subsequences. "
        "Rolling out past that horizon in imagination accumulates errors and the frames start "
        "to look nothing like real MiniGrid states, which further hurts the CLIP scorer.",
        "<b>CLIP prompt sensitivity.</b> Small changes to the prompts ("
        "&ldquo;agent on the goal&rdquo; vs &ldquo;red triangle on green goal&rdquo;) shift the "
        "score distribution significantly. A contrastive pair (positive − negative) is much more "
        "stable than a single-prompt cosine similarity, but the choice of the <i>negative</i> "
        "prompt matters too.",
        "<b>Reward head is very sparse.</b> Because the only positive reward is at the goal and "
        "random data rarely reaches the goal in 8×8 Empty, the RSSM's reward head is close to "
        "constant. The <i>wm_reward</i> baseline therefore behaves near-randomly, and the VLM "
        "signal is the primary source of directed exploration in this setup.",
        "<b>MiniGrid rendering vs. natural images.</b> CLIP is trained on natural photographs; "
        "MiniGrid renders are cartoon-like and out-of-distribution. Score gradients along a "
        "trajectory exist but are small, which is why we need a contrastive pair and a discount "
        "over the horizon rather than picking a single frame.",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    story.append(Paragraph("4.2 What we would try with more time", styles["h3"]))
    for b in [
        "<b>Longer training with more diverse data.</b> Mix random + planner-collected rollouts "
        "(Dreamer's data-collection loop) and train the RSSM for 100k+ updates. Reward-head "
        "accuracy in particular should improve dramatically once the buffer contains successful "
        "trajectories.",
        "<b>CEM over random shooting.</b> Replace random shooting with a small CEM (e.g. 3 "
        "iterations, top-k=10% elites, 100 candidates). Same interface, better sample "
        "efficiency in the planner.",
        "<b>Actor-critic on top of the world model.</b> The full Dreamer objective (imagined "
        "policy trained on λ-returns of a value function) would remove the per-step MPC cost "
        "and produce a stronger baseline.",
        "<b>Better VLM.</b> SigLIP tends to be a stronger scorer on agent-centric captions, "
        "and DINO/CLIPSeg / Grounded-SAM provide dense goal maps rather than a single scalar. "
        "Alternatively, distill CLIP scores into a small MLP head trained on decoded frames to "
        "avoid running CLIP inside the planner loop (huge speed-up).",
        "<b>Score real frames too.</b> A useful ablation: apply the VLM to the real observation "
        "at each step as a shaping bonus, so the RSSM reward head can learn a dense pseudo-reward. "
        "This decouples the world-model quality from the VLM's usefulness for planning.",
        "<b>Harder environments.</b> Move to DoorKey / KeyCorridor with multi-clause prompts "
        "(&ldquo;agent next to the key&rdquo; → &ldquo;agent in front of the door&rdquo;) to test whether "
        "CLIP can guide subgoal switching.",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    # ---- Refs ----
    story.append(Paragraph("References", styles["h2"]))
    story.append(Paragraph(
        "1. Hafner, D. et al. <b>Learning Latent Dynamics for Planning from Pixels</b> (PlaNet). "
        "<a href='https://arxiv.org/abs/1811.04551' color='#20808D'>arXiv:1811.04551</a>",
        styles["body"]))
    story.append(Paragraph(
        "2. Hafner, D. et al. <b>Mastering Diverse Domains through World Models</b> (DreamerV3). "
        "<a href='https://arxiv.org/abs/2301.04104' color='#20808D'>arXiv:2301.04104</a>",
        styles["body"]))
    story.append(Paragraph(
        "3. Radford, A. et al. <b>Learning Transferable Visual Models from Natural Language Supervision</b> (CLIP). "
        "<a href='https://arxiv.org/abs/2103.00020' color='#20808D'>arXiv:2103.00020</a>",
        styles["body"]))
    story.append(Paragraph(
        "4. Chevalier-Boisvert, M. et al. <b>MiniGrid & Miniworld</b>. "
        "<a href='https://minigrid.farama.org/' color='#20808D'>minigrid.farama.org</a>",
        styles["body"]))

    doc.build(story)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
