"""
Animated visualization of the Agentic RAG pipeline.

Run with:
    python pipeline_animation.py

Saves pipeline_animation.gif in the project root.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation

BG_COLOR   = "#0f1117"
CARD_COLOR = "#1e2130"
HIGHLIGHT  = "#f7c948"
COLORS = {
    "ingest":   "#4c9be8",
    "embed":    "#7b61ff",
    "retrieve": "#00c4a7",
    "rerank":   "#ff6b6b",
    "agent":    "#f7c948",
    "llm":      "#ff9f43",
    "answer":   "#a8e063",
}

STAGES = [
    ("PDF\nIngestion",      "PyPDF + chunking\n800 tok / 120 overlap",          1.1,  5.5, "ingest"),
    ("Embedding",           "BAAI/bge-small-en-v1.5\nsentence-transformers",    3.4,  5.5, "embed"),
    ("Chroma\nVector DB",   "Persisted locally",                                5.7,  5.5, "embed"),
    ("Hybrid\nRetrieval",   "Dense 60% + BM25 40%\ntop-15 each",               8.0,  5.5, "retrieve"),
    ("Cross-Encoder\nRerank","ms-marco-MiniLM-L-6-v2\ntop-5 chunks",           10.3, 5.5, "rerank"),
    ("Router /\nSelf-Check","RETRIEVE or DIRECT?\nSufficiency check + rewrite", 8.0,  2.8, "agent"),
    ("HF LLM",              "Qwen2.5-7B-Instruct-Turbo\nInference API",         10.3, 2.8, "llm"),
    ("Answer +\nSources",   "Grounded reply\nwith citations",                   12.5, 2.8, "answer"),
]

STRAIGHT_ARROWS = [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7)]
TOTAL_FRAMES = 130
FADE         = 12


def build_frame(ax, alpha_cards, alpha_arrows, alpha_loop, alpha_title, alpha_query):
    ax.cla()
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 14)
    ax.set_ylim(1.5, 7.2)
    ax.axis("off")

    # Title
    ax.text(7, 6.85, "Agentic RAG Pipeline  --  Hugging Face Edition",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", alpha=alpha_title)

    # Query badge
    ax.text(1.1, 6.4, 'User Query: "What is RAG?"',
            ha="center", va="center", fontsize=7.5, color=HIGHLIGHT,
            alpha=alpha_query,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e1a08",
                      edgecolor=HIGHLIGHT, linewidth=1.2, alpha=alpha_query))

    # Section labels
    ax.text(5.7, 6.55, "[ Indexing Pipeline ]",
            ha="center", fontsize=7, color="#555577", style="italic",
            alpha=min(alpha_cards[2], 1.0))
    ax.text(10.3, 1.9, "[ Inference Pipeline ]",
            ha="center", fontsize=7, color="#555577", style="italic",
            alpha=min(alpha_cards[6], 1.0))

    # Cards
    for i, (label, sub, x, y, ck) in enumerate(STAGES):
        a = alpha_cards[i]
        c = COLORS[ck]
        rect = mpatches.FancyBboxPatch(
            (x - 0.95, y - 0.58), 1.9, 1.16,
            boxstyle="round,pad=0.08",
            facecolor=CARD_COLOR, edgecolor=c, linewidth=2, alpha=a, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y + 0.15, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color=c, alpha=a,
                zorder=4, multialignment="center")
        ax.text(x, y - 0.25, sub, ha="center", va="center",
                fontsize=6.2, color="#aaaacc", alpha=a,
                zorder=4, multialignment="center")

    # Straight arrows
    for idx, (fi, ti) in enumerate(STRAIGHT_ARROWS):
        a = alpha_arrows[idx]
        _, _, x0, y0, ck = STAGES[fi]
        _, _, x1, y1, _  = STAGES[ti]
        ax.add_patch(FancyArrowPatch(
            (x0 + 0.96, y0), (x1 - 0.96, y1),
            arrowstyle="-|>", connectionstyle="arc3,rad=0.0",
            color=COLORS[ck], linewidth=1.6, mutation_scale=13,
            alpha=a, zorder=2))

    # Rerank -> Router (down curve)
    ax.add_patch(FancyArrowPatch(
        (10.3, 4.92), (8.0 + 0.96, 2.8 + 0.58),
        arrowstyle="-|>", connectionstyle="arc3,rad=-0.35",
        color=COLORS["rerank"], linewidth=1.6, mutation_scale=13,
        alpha=alpha_arrows[6], zorder=2))

    # Router -> Retrieval (query-rewrite dashed loop)
    ax.add_patch(FancyArrowPatch(
        (8.0 - 0.96, 2.8 + 0.58), (8.0 - 0.96, 5.5 - 0.58),
        arrowstyle="-|>", connectionstyle="arc3,rad=0.5",
        color=COLORS["agent"], linewidth=1.6, linestyle=(0, (4, 3)),
        mutation_scale=13, alpha=alpha_loop, zorder=2))
    ax.text(6.55, 4.15, "query\nrewrite",
            ha="center", fontsize=6, color=COLORS["agent"],
            style="italic", alpha=alpha_loop)


def get_alpha(frame, start):
    return min(1.0, max(0.0, (frame - start) / FADE))


def make_schedule():
    # Returns list of (element_index, start_frame)
    n = len(STAGES) + len(STRAIGHT_ARROWS) + 2 + 3  # +2 loop/down, +3 title/query/sections
    step = TOTAL_FRAMES / n
    sched = []
    t = 0
    sched.append(("title", int(t))); t += step
    sched.append(("query", int(t))); t += step
    for i in range(len(STAGES)):
        sched.append((f"card{i}", int(t))); t += step
    for i in range(len(STRAIGHT_ARROWS)):
        sched.append((f"arr{i}", int(t))); t += step
    sched.append(("arr_down", int(t))); t += step
    sched.append(("arr_loop", int(t)))
    return {k: v for k, v in sched}


schedule = make_schedule()

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG_COLOR)


def animate(frame):
    alpha_title = get_alpha(frame, schedule["title"])
    alpha_query = get_alpha(frame, schedule["query"])
    alpha_cards  = [get_alpha(frame, schedule[f"card{i}"]) for i in range(len(STAGES))]
    alpha_arrows = [get_alpha(frame, schedule[f"arr{i}"]) for i in range(len(STRAIGHT_ARROWS))]
    alpha_arrows.append(get_alpha(frame, schedule["arr_down"]))
    alpha_loop   = get_alpha(frame, schedule["arr_loop"])
    build_frame(ax, alpha_cards, alpha_arrows, alpha_loop, alpha_title, alpha_query)


anim = FuncAnimation(fig, animate, frames=TOTAL_FRAMES,
                     interval=60, blit=False, repeat=True)

plt.tight_layout(pad=0.5)
anim.save("pipeline_animation.gif", writer="pillow", fps=18, dpi=110)
print("Saved: pipeline_animation.gif")
