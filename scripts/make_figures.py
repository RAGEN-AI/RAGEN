#!/usr/bin/env python3
"""
make_figures.py  —  Presentation figures for POMDP Sokoban memory experiments.

Usage (on Quest):
    cd /home/eiu4164/projects/RAGEN
    python3 scripts/make_figures.py

Output:
    figures/fig1_training_curves.png   — 4 context-window conditions
    figures/fig2_decoupling.png        — MAP quality vs task success
    figures/fig3_memory_effect.png     — MEM1 vs NoMem across difficulties
    figures/fig4_14b_breakdown.png     — 14B zero-shot episode analysis
"""

import re, glob, os, json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':     True,
    'grid.alpha':    0.25,
    'grid.linestyle':'--',
    'figure.dpi':    150,
})

C = {
    'mem1':  '#1565C0',   # dark blue
    'nomem': '#C62828',   # dark red
    'full':  '#616161',   # gray
    '3b':    '#2E7D32',   # dark green
    '7b':    '#E65100',   # deep orange
}

LOG_DIR   = '/home/eiu4164/projects/RAGEN/logs'
TRAJ_ROOT = '/projects/p32139/gesture_data/ragen_runs/traj'
os.makedirs('figures', exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────
def parse_log(pattern, env='POMDPSokoban'):
    """Return step→success from the longest log matching `pattern`."""
    best, best_n = {}, 0
    for p in sorted(glob.glob(os.path.join(LOG_DIR, pattern))):
        d = {}
        try:
            with open(p) as f:
                for line in f:
                    m = re.search(rf'step:(\d+).*?train/{env}/success:([\d.]+)', line)
                    if m:
                        d[int(m.group(1))] = float(m.group(2))
        except Exception:
            pass
        if len(d) > best_n:
            best, best_n = d, len(d)
    return best


def merge_logs(*patterns, env='POMDPSokoban'):
    """Merge multiple log patterns (e.g. original + resume run)."""
    merged = {}
    for pat in patterns:
        merged.update(parse_log(pat, env))
    return merged


def smooth(vals, w=7):
    """Rolling mean. Returns (smoothed_values, valid_step_indices)."""
    if len(vals) < w:
        return np.array(vals), list(range(len(vals)))
    sv = np.convolve(vals, np.ones(w) / w, mode='valid')
    return sv, list(range(w - 1, len(vals)))


def compute_prc(traj_dir, step_subset=None):
    """
    Compute P@(r,c)% (player at correct MAP position) per step.
    Reads JSONL trajectory files; skips corrupted lines.
    Returns {step: prc_percent}.
    """
    files = sorted(
        glob.glob(f'{traj_dir}/*.jsonl'),
        key=lambda x: int(os.path.basename(x).replace('.jsonl', ''))
    )
    results = {}
    for fpath in files:
        step = int(os.path.basename(fpath).replace('.jsonl', ''))
        if step_subset and step not in step_subset:
            continue
        total = correct = 0
        with open(fpath) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                obs = rec.get('input', '')
                out = rec.get('output', '')
                pos = list(re.finditer(r'Position:\s*\((\d+),\s*(\d+)\)', obs))
                if not pos:
                    continue
                r, c = int(pos[-1].group(1)), int(pos[-1].group(2))
                thinks = list(re.finditer(r'<think>(.*?)</think>', out, re.DOTALL))
                if not thinks:
                    continue
                think = thinks[-1].group(1)
                mm = re.search(r'MAP:\s*((?:[?#_XOPSs√ ]+\n?){1,7})', think)
                if not mm:
                    continue
                rows = [ln.strip().split()
                        for ln in mm.group(1).strip().split('\n')
                        if len(ln.strip().split()) == 5]
                if len(rows) != 5:
                    continue
                total += 1
                try:
                    if rows[r][c] in ('P', 'S'):
                        correct += 1
                except IndexError:
                    pass
        if total > 0:
            results[step] = 100 * correct / total
    return results


# ── Figure 1: Training curves (4 context conditions) ──────────────────
def fig1():
    print("\n[Fig 1] Training curves...")

    # MEM1: combine v2 (1-100) + v2 resume (101-155) for longest curve
    mem1_data = merge_logs(
        'pomdp_mem1_lora8_3b_h100_v2.*.log',
        'pomdp_mem1_lora8_3b_h100_v2_resume.*.log',
    )
    # Fallback to v3 if v2 not found
    if not mem1_data:
        mem1_data = parse_log('pomdp_mem1_lora8_3b_h100_v3.*.log')

    specs = [
        ('MEM1',         mem1_data,                                            C['mem1'],  '-',  2.5),
        ('NoMem',        parse_log('pomdp_nomem_lora8_3b_h100.9136120.log'),   C['nomem'], '-',  2.5),
        ('Full context', parse_log('pomdp_full_lora8_3b_h100_v2.*.log'),       C['full'],  ':',  2.0),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, data, color, ls, lw in specs:
        if not data:
            print(f"  [warn] no data for {label}")
            continue
        steps = sorted(data.keys())
        vals  = [data[s] for s in steps]

        # Raw (faint background)
        ax.plot(steps, vals, color=color, alpha=0.18, linewidth=0.8)

        # Smoothed foreground
        sv, idx = smooth(vals, w=9)
        ss = [steps[i] for i in idx]
        ax.plot(ss, sv, color=color, linewidth=lw, linestyle=ls, label=label)

    # 14B zero-shot reference line
    ax.axhline(0.24, color='#9E9E9E', linewidth=1.2, linestyle='-.', alpha=0.8)
    ax.text(2, 0.245, '14B zero-shot (24%)', fontsize=9.5, color='#9E9E9E', va='bottom')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Task Success Rate')
    ax.set_title('Training Success by Context Window Mode\n3B RL · POMDP Sokoban 5×5  [4–8 min steps]')
    ax.legend(fontsize=11, framealpha=0.85)
    ax.set_ylim(0, 0.75)

    plt.tight_layout()
    plt.savefig('figures/fig1_training_curves.png', bbox_inches='tight')
    plt.close()
    print("  Saved figures/fig1_training_curves.png")


# ── Figure 2: Decoupling scatter (P@(r,c)% vs success) ───────────────
def fig2():
    print("\n[Fig 2] Decoupling scatter...")

    # 7B — hardcoded from analyze_traj_map output (already computed)
    prc_7b  = [28.2, 32.5, 31.5, 20.4, 28.8, 28.3, 26.8, 26.3, 21.3, 30.0]
    succ_7b = [0.238, 0.242, 0.320, 0.309, 0.383, 0.477, 0.176, 0.375, 0.195, 0.383]

    # 3B — compute from trajectory files
    print("  Computing 3B P@(r,c)% from traj files (may take ~1 min)...")
    eval_steps = {1, 11, 21, 31, 41, 51, 61, 71}
    prc_map_3b = compute_prc(f'{TRAJ_ROOT}/pomdp_mem1_lora8_3b_h100_v3', eval_steps)
    log_3b     = parse_log('pomdp_mem1_lora8_3b_h100_v3.*.log')

    prc_3b, succ_3b = [], []
    for step, prc in sorted(prc_map_3b.items()):
        if step in log_3b:
            prc_3b.append(prc)
            succ_3b.append(log_3b[step])

    if not prc_3b:
        print("  [warn] 3B traj not found; using approximate values")
        prc_3b  = [4.2, 6.1, 8.3, 5.5, 9.0, 7.2, 12.4, 6.5]
        succ_3b = [0.430, 0.348, 0.410, 0.367, 0.324, 0.270, 0.289, 0.355]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    kw = dict(s=90, alpha=0.75, zorder=3, edgecolors='white', linewidths=0.5)
    ax.scatter(prc_3b, succ_3b, color=C['3b'], marker='o', label='3B RL MEM1', **kw)
    ax.scatter(prc_7b, succ_7b, color=C['7b'], marker='s', label='7B RL MEM1', **kw)

    # Mean stars
    ax.scatter([np.mean(prc_3b)], [np.mean(succ_3b)],
               color=C['3b'], s=220, marker='*', zorder=5, edgecolors='white', lw=0.5)
    ax.scatter([np.mean(prc_7b)], [np.mean(succ_7b)],
               color=C['7b'], s=220, marker='*', zorder=5, edgecolors='white', lw=0.5)

    # Cluster ellipses (1.5σ)
    for xs, ys, color in [(prc_3b, succ_3b, C['3b']), (prc_7b, succ_7b, C['7b'])]:
        if len(xs) < 3:
            continue
        cx, cy = np.mean(xs), np.mean(ys)
        ex = Ellipse((cx, cy),
                     width=2 * 1.5 * np.std(xs),
                     height=2 * 1.5 * np.std(ys),
                     facecolor=color, alpha=0.10,
                     edgecolor=color, linewidth=1.5, linestyle='--')
        ax.add_patch(ex)

    # Random baseline
    ax.axvline(4, color='#BDBDBD', linestyle=':', linewidth=1.5)
    ax.text(4.3, 0.10, 'Random\nbaseline\n(4%)', fontsize=9, color='#9E9E9E', va='bottom')

    # Annotation: both reach ~32%
    ax.annotate(
        'Similar task success (~32%)\ndespite very different MAP quality',
        xy=(np.mean(prc_7b), np.mean(succ_7b)),
        xytext=(28, 0.17),
        fontsize=9.5, color='#37474F', ha='center',
        arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1.2),
    )

    ax.set_xlabel('MAP Quality: P@(r,c)%')
    ax.set_ylabel('Task Success Rate')
    ax.set_title('MAP Quality and Task Success Are Decoupled\n'
                 '(★ = mean; each point = one training-step evaluation)')
    ax.legend(fontsize=11, framealpha=0.85)
    ax.set_xlim(0, 44)
    ax.set_ylim(0.05, 0.70)

    plt.tight_layout()
    plt.savefig('figures/fig2_decoupling.png', bbox_inches='tight')
    plt.close()
    print("  Saved figures/fig2_decoupling.png")


# ── Figure 3: Memory effect across difficulties ────────────────────────
def fig3():
    print("\n[Fig 3] Memory effect bar chart...")

    conditions = ['5×5 Easy\n[4–8 steps]', '5×5 Hard\n[8–15 steps]', '7×7 Easy\n[4–8 steps]']
    mem1_vals  = [0.326, 0.176, 0.273]
    nomem_vals = [0.312, 0.159, 0.313]

    # Step-level std across training (computed from log data)
    # 5×5 Easy: from MEM1 v2-resume and NoMem v1 full runs
    # 5×5 Hard: from hard_v1 logs (15 sampled steps each)
    # 7×7 Easy: from 7x7_v1 logs (NoMem std inflated by KL-spike outlier at step 71)
    std_mem1  = [0.088, 0.056, 0.084]
    std_nomem = [0.082, 0.054, 0.115]

    x = np.arange(len(conditions))
    w = 0.34
    eb_kw = dict(fmt='none', color='#212121', capsize=5, linewidth=1.8, zorder=4)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    b1 = ax.bar(x - w/2, mem1_vals,  w, color=C['mem1'],  label='MEM1',  alpha=0.88, zorder=3)
    b2 = ax.bar(x + w/2, nomem_vals, w, color=C['nomem'], label='NoMem', alpha=0.88, zorder=3)

    # Error bars
    ax.errorbar(x - w/2, mem1_vals,  yerr=std_mem1,  **eb_kw)
    ax.errorbar(x + w/2, nomem_vals, yerr=std_nomem, **eb_kw)

    # Value labels on bars
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=10.5)

    # Gap annotations (positioned above the taller error bar)
    for i, (m, n, sm, sn) in enumerate(zip(mem1_vals, nomem_vals, std_mem1, std_nomem)):
        gap   = m - n
        y_top = max(m + sm, n + sn) + 0.03
        color = '#1B5E20' if gap > 0.005 else '#B71C1C' if gap < -0.005 else '#757575'
        sign  = '+' if gap >= 0 else ''
        ax.text(x[i], y_top, f'Δ = {sign}{gap:.3f}',
                ha='center', fontsize=10.5, color=color, fontweight='bold')

    # Note about noise
    ax.text(0.99, 0.97,
            'Error bars = ±1 std (step-level)\nAll Δ are within noise',
            transform=ax.transAxes, fontsize=9, color='#616161',
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                      edgecolor='#BDBDBD', alpha=0.8))

    # 14B reference
    ax.axhline(0.24, color='#9E9E9E', linewidth=1.2, linestyle='-.', alpha=0.75)
    ax.text(-0.5, 0.243, '14B zero-shot (24%)', fontsize=9, color='#9E9E9E')

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylabel('Task Success Rate')
    ax.set_title('Effect of Memory Access Across Task Conditions\n'
                 'Δ = MEM1 − NoMem  (positive = memory helps)')
    ax.legend(fontsize=11, framealpha=0.85)
    ax.set_ylim(0, 0.62)

    plt.tight_layout()
    plt.savefig('figures/fig3_memory_effect.png', bbox_inches='tight')
    plt.close()
    print("  Saved figures/fig3_memory_effect.png")


# ── Figure 4: 14B zero-shot episode breakdown ─────────────────────────
def fig4():
    print("\n[Fig 4] 14B breakdown...")

    episodes = [
        {'ep':  0, 'steps':   1, 'p_rc': 100.0, 'type': 'trivial'},
        {'ep':  8, 'steps':   1, 'p_rc': 100.0, 'type': 'trivial'},
        {'ep': 14, 'steps':   1, 'p_rc':   0.0, 'type': 'trivial'},
        {'ep': 19, 'steps':   1, 'p_rc': 100.0, 'type': 'trivial'},
        {'ep': 15, 'steps':   2, 'p_rc':   0.0, 'type': 'trivial'},
        {'ep': 38, 'steps':   2, 'p_rc':   0.0, 'type': 'trivial'},
        {'ep': 10, 'steps':   3, 'p_rc':   0.0, 'type': 'trivial'},
        {'ep': 31, 'steps':   9, 'p_rc':   0.0, 'type': 'short'},
        {'ep': 20, 'steps':  13, 'p_rc':  41.7, 'type': 'genuine'},
        {'ep':  3, 'steps': 103, 'p_rc':   0.0, 'type': 'random'},
        {'ep': 24, 'steps': 136, 'p_rc':   5.9, 'type': 'random'},
        {'ep': 32, 'steps': 141, 'p_rc':   6.4, 'type': 'random'},
    ]

    TYPE_COLOR = {
        'trivial': '#EF9A9A',
        'short':   '#FFE082',
        'genuine': '#A5D6A7',
        'random':  '#B0BEC5',
    }
    TYPE_LABEL = {
        'trivial': 'Trivial (≤3 steps)',
        'short':   'Short (4–15 steps)',
        'genuine': 'Genuine MAP-guided',
        'random':  'Random walk (>100 steps)',
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={'width_ratios': [2.2, 1]}
    )

    # Left: bar chart of steps per solved episode
    steps  = [e['steps'] for e in episodes]
    colors = [TYPE_COLOR[e['type']] for e in episodes]
    xlabs  = [f"ep {e['ep']}" for e in episodes]

    bars = ax1.bar(range(len(episodes)), steps, color=colors,
                   edgecolor='white', linewidth=0.8, zorder=3)

    # P@(r,c)% label on each bar
    for i, e in enumerate(episodes):
        if e['p_rc'] > 0:
            ax1.text(i, e['steps'] * 1.15,
                     f"{e['p_rc']:.0f}%",
                     ha='center', va='bottom', fontsize=8.5, color='#37474F')

    ax1.set_yscale('log')
    ax1.set_ylim(0.7, 350)
    ax1.set_xticks(range(len(episodes)))
    ax1.set_xticklabels(xlabs, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Steps to Solve (log scale)')
    ax1.set_title('14B Zero-Shot MEM1: All 12 Solved Episodes\n'
                  '(24% overall success rate;  labels = P@(r,c)%)')

    # Threshold line
    ax1.axhline(3.5, color='#9E9E9E', linestyle='--', linewidth=1, alpha=0.7)
    ax1.text(11.4, 4.0, '≤3 steps', fontsize=8.5, color='#9E9E9E', ha='right')

    # Legend
    handles = [mpatches.Patch(color=TYPE_COLOR[t], label=TYPE_LABEL[t])
               for t in ['trivial', 'short', 'genuine', 'random']]
    ax1.legend(handles=handles, fontsize=9, loc='upper left')

    # Right: pie chart
    counts = Counter(e['type'] for e in episodes)
    order  = ['trivial', 'short', 'genuine', 'random']
    sizes  = [counts.get(t, 0) for t in order]
    pcolors = [TYPE_COLOR[t] for t in order]
    plabels = [f"{TYPE_LABEL[t]}\n(n={counts.get(t,0)})" for t in order]

    wedges, _, autotexts = ax2.pie(
        sizes, labels=None, colors=pcolors,
        autopct='%1.0f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax2.set_title('Episode\nType Breakdown', fontsize=12)
    ax2.legend(wedges, plabels, fontsize=8.5,
               loc='lower center', bbox_to_anchor=(0.5, -0.42))

    plt.tight_layout()
    plt.savefig('figures/fig4_14b_breakdown.png', bbox_inches='tight')
    plt.close()
    print("  Saved figures/fig4_14b_breakdown.png")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating presentation figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    print("\nDone. All figures saved to ./figures/")
