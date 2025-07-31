# [markdown]
#  Input

FILE = "../Verdict/data/stats/stats.csv"
TRENDS_VARIABLES = [
    ("num_mb", 5),
    ("num_tp", 4),
    ("num_dp", 5),
    ("num_layers", 4),
    ("num_pp", 4), 
    ("gbs", 4),
    ("num_heads", 5),
    ("hidden_size", 4),
    ("seqlen", 5),
]


# [markdown]
#  Definition
# Plot Functions

import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.ticker as ticker
from  matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import os, shutil

figs_dir = "figs"
if os.path.exists(figs_dir):
    shutil.rmtree(figs_dir)
os.mkdir(figs_dir)


@dataclass
class PlotConfig:
    figsize = (8,6)
    font_big = 30
    font_small = 35
    font_legend = 15

def draw_lines(data: Dict[str,List], 
               xlabel: str = "", 
               ylabel: str = "",
               highlight_idxs: List = None,
               ylim: Tuple[int] = None,
               
               export_fname: str = "",
               show: bool = False,
               cfg: PlotConfig = None):
    
    highlight_idxs = highlight_idxs or []
    cfg  = cfg or PlotConfig()
    
    # data
    df = pd.DataFrame(data)
    plt.figure(figsize=cfg.figsize)
    plt.plot(df['X'], df['Y'], marker='o', color='darkcyan', linestyle='-', linewidth=2, markersize=8)
    
    # label
    if xlabel:
        plt.xlabel(xlabel, fontsize=cfg.font_big)
    if ylabel:
        plt.ylabel(ylabel, fontsize=cfg.font_big)
        
    # ticks
    plt.xticks(df['X'], fontsize=cfg.font_small)
    plt.yticks(fontsize=cfg.font_small)
    
    # range
    if ylim:
        plt.ylim(*ylim)
    
    # highlights
    for i in highlight_idxs:
        plt.scatter(df['X'][i], df['Y'][i], color='orange', zorder=5, marker='s', s=100)

    # export
    plt.tight_layout()
    if export_fname:
        plt.savefig(f"figs/{export_fname}.pdf", format="pdf", transparent=True, bbox_inches='tight', pad_inches=0)
    
    # display
    if show:
        print(export_fname)
        plt.show()




def draw_hybrid_bar_line(data: Dict[str, List], 
         xlabel: str = "", 
         ylabel: str = "",
         highlight_idxs: List = None,
         ylim: Tuple[int] = None,
         export_fname: str = "",
         show: bool = False,
         cfg: PlotConfig = None,
         legend: bool = False,
         uniform_spacing: bool = False,
         component_labels: List[str] = None):

    highlight_idxs = highlight_idxs or []
    cfg = cfg or PlotConfig()

    x = data["X"]
    positions = list(range(len(x))) if uniform_spacing else x
    components = data["Y"]  # shape: [component][x_index]
    num_components = len(components)

    if component_labels is None:
        component_labels = [f'Component {i+1}' for i in range(num_components)]

    default_colors = ["264653","2a9d8f","e9c46a","e76f51","f4a261",]
    default_colors = [f"#{c}" for c in default_colors]
    assert len(default_colors) >= num_components, f"not enough colors"
    
    fig, ax = plt.subplots(figsize=cfg.figsize)
    
    # xlim
    if uniform_spacing:
        ax.set_xlim(-0.5, len(x) - 0.5)
    else:
        XLIM_ALPHA = 1 / (len(x)+3)
        xrange = max(x)-min(x) + 1
        xlim_margin = xrange * XLIM_ALPHA
        ax.set_xlim(min(0, min(x)-1), max(x) + xlim_margin)
        # ax.set_xlim(min(x) - 2, max(x) + xlim_margin)
        # ax.set_xlim(None, max(x) + xlim_margin)

    # Bar Width
    ABS_BAR_WIDTH = 0.4
    # Get figure size and axis position
    fig_width_inch = fig.get_size_inches()[0]
    xlim = ax.get_xlim()
    xrange = xlim[1] - xlim[0]
    # Compute bar width in data units
    bar_width = (ABS_BAR_WIDTH / fig_width_inch) * xrange
    
    # Bar plot (stacked components)
    EPS = 0
    bottom = [EPS] * len(x)
    for comp_vals, color, label in zip(components, default_colors, component_labels):
        ax.bar(positions, comp_vals, bottom=bottom, width=bar_width, label=label, color=color, zorder=2)
        bottom = [b + v for b, v in zip(bottom, comp_vals)]

    # Line plot (total)
    total_vals = [sum(comp.iloc[i] for comp in components) for i in range(len(x))]
    # ax.plot(positions, total_vals, marker='o', color='black', linewidth=1.5, label='Total', zorder=3)

    # Highlight selected indices (optional)
    for i in highlight_idxs:
        ax.scatter(positions[i], total_vals[i], color='black', s=100, zorder=4, edgecolor='black', marker='s')

    # Labels and ticks
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=cfg.font_small)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=cfg.font_small)

    ax.set_xticks(positions)
    ax.set_xticklabels(x, fontsize=cfg.font_small)
    ax.tick_params(axis='y', labelsize=cfg.font_small)

    # ylim
    YLIM_ALPHA = 1.3
    if ylim:
        plt.ylim(*ylim)
    else:
        ax.set_ylim(None, YLIM_ALPHA*max(total_vals))
        
    # legend
    if legend:
        ax.legend(fontsize=cfg.font_legend)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, zorder=1)
        
    plt.tight_layout()
    
    if export_fname:
        plt.savefig(f"figs/{export_fname}.pdf", format="pdf", transparent=True, bbox_inches='tight', pad_inches=0)
    if show:
        print(export_fname)
        plt.show()


def draw_hybrid(data: Dict[str, List], 
         xlabel: str = "", 
         ylabel: str = "",
         highlight_idxs: List = None,
         ylim: Tuple[int] = None,
         export_fname: str = "",
         show: bool = False,
         cfg: PlotConfig = None,
         legend: bool = False,
         uniform_spacing: bool = False,
         log_y_ticks: bool = False,
         line_width: float = None,
         component_labels: List[str] = None):

    highlight_idxs = highlight_idxs or []
    cfg = cfg or PlotConfig()

    x = data["X"]
    positions = list(range(len(x))) if uniform_spacing else x
    components = data["Y"]  # shape: [component][x_index]
    num_components = len(components)

    if component_labels is None:
        component_labels = [f'Component {i+1}' for i in range(num_components)]

    default_colors = ["264653","2a9d8f","e9c46a","e76f51","f4a261",]
    default_colors = [f"#{c}" for c in default_colors]
    assert len(default_colors) >= num_components, f"not enough colors"
    
    fig, ax = plt.subplots(figsize=cfg.figsize)
    
    # xlim
    if uniform_spacing:
        ax.set_xlim(-0.5, len(x) - 0.5)
    else:
        XLIM_ALPHA = 1 / (len(x)+3)
        xrange = max(x)-min(x) + 1
        xlim_margin = xrange * XLIM_ALPHA
        ax.set_xlim(min(0, min(x)-1), max(x) + xlim_margin)
        # ax.set_xlim(min(x) - 2, max(x) + xlim_margin)
        # ax.set_xlim(None, max(x) + xlim_margin)

    # Bar Width
    ABS_BAR_WIDTH = 0.4
    # Get figure size and axis position
    fig_width_inch = fig.get_size_inches()[0]
    xlim = ax.get_xlim()
    xrange = xlim[1] - xlim[0]
    # Compute bar width in data units
    bar_width = (ABS_BAR_WIDTH / fig_width_inch) * xrange
    
    # Bar plot (stacked components)
    # EPS = 1e-4 if log_y_ticks else 0
    # bottom = [EPS] * len(x)
    # for comp_vals, color, label in zip(components, default_colors, component_labels):
    #     if log_y_ticks:
    #         comp_vals = np.clip(comp_vals, EPS, None)
    #     ax.bar(positions, comp_vals, bottom=bottom, width=bar_width, label=label, color=color, zorder=2)
    #     bottom = [b + v for b, v in zip(bottom, comp_vals)]
    
    num_components = len(components)
    group_width = 0.5  # portion of space at each tick allocated to bars
    bar_width = group_width / num_components

    # Adjust positions per component
    for i, (comp_vals, color, label) in enumerate(zip(components, default_colors, component_labels)):
        if log_y_ticks:
            comp_vals = np.clip(comp_vals, 1e-4, None)
        offset = (i - (num_components - 1) / 2) * bar_width  # center the group
        bar_positions = [p + offset for p in positions]
        ax.bar(bar_positions, comp_vals, width=bar_width, label=label, color=color, zorder=2)

    # Line plot (total)
    total_vals = [sum(comp.iloc[i] for comp in components) for i in range(len(x))]
    if line_width is None:
        line_width = 1.5 
    ax.plot(positions, total_vals, marker='o', color='black', linewidth=line_width, label='Total', zorder=3)

    # Highlight selected indices (optional)
    for i in highlight_idxs:
        ax.scatter(positions[i], total_vals[i], color='black', s=100, zorder=4, edgecolor='black', marker='s')

    # Labels and ticks
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=cfg.font_small)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=cfg.font_small)

    ax.set_xticks(positions)
    ax.set_xticklabels(x, fontsize=cfg.font_small)
    ax.tick_params(axis='y', labelsize=cfg.font_small)

    # ylim
    YLIM_ALPHA = 1.3
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(None, YLIM_ALPHA*max(total_vals))
        
    # legend
    if legend:
        ax.legend(fontsize=cfg.font_legend)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, zorder=1)
        
    if log_y_ticks:
        ax.set_yscale("log", base=2)
        min_val = min([min(comp) for comp in components if len(comp) > 0])
        min_display_val = max(1, min_val * 0.5)
        ax.set_ylim(bottom=min_display_val)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        
        
        
    plt.tight_layout()
    
    if export_fname:
        plt.savefig(f"figs/{export_fname}.pdf", format="pdf", transparent=True, bbox_inches='tight', pad_inches=0)
    if show:
        print(export_fname)
        plt.show()

# [markdown]
# Data Preparation

# sanity check
df_full = pd.read_csv(FILE)
df_num_rows = len(df_full)
expected_num_rows = sum([cnt for _,cnt in TRENDS_VARIABLES])
assert df_num_rows == expected_num_rows, f"dataframe has inconsistent number of rows: {df_num_rows} != {expected_num_rows}."

def get_var_slc_range(var: str):
    start = 0
    end = 0
    assigned = False
    for name, cnt in TRENDS_VARIABLES:
        if name == var:
            assigned = True
            end = start + cnt
            break
        else:
            start += cnt
    assert assigned, f"{var} not in valid variable names: {[name for name,_ in TRENDS_VARIABLES]}"
    return (start, end)

def get_var_df(var: str):
    start, end = get_var_slc_range(var)
    return df_full[start:end]
    

# [markdown]
# Plot Variable Trends

import numpy as np
import pandas as pd
from typing import Dict, List, Union

def log_scale_stacked_data(
    data: Dict[str, List[Union[int, float]]], 
    eps: float = 1e-9,
) -> Dict[str, List]:
    """
    Transform Y components into log2(total) scaling with segment-wise proportionality.
    Also formats X values as LaTeX-style superscripts (e.g., $2^{9}$).

    Args:
        data: Dict with 'X' (Series or list of ints) and 'Y' (list of pd.Series or list-like)
        eps: Small value to avoid log(0)

    Returns:
        New dict with:
          - 'X': pd.Series of str like '$2^{n}$' for matplotlib rendering
          - 'Y': List of pd.Series (log-scaled, proportional)
    """
    # Convert Y components to array
    raw_components = np.vstack([np.array(y) for y in data["Y"]])  # shape: [num_components, num_bars]
    total = np.sum(raw_components, axis=0)
    log_total = np.log2(np.clip(total, eps, None))

    # Compute proportionally scaled values
    normalized = raw_components / total
    log_scaled = normalized * log_total

    # Convert Y components back to Series
    scaled_components = [pd.Series(log_scaled[i, :]) for i in range(log_scaled.shape[0])]

    # Convert X to LaTeX superscript strings
    x_values = pd.Series(data["X"])
    # x_formatted = x_values.apply(lambda v: rf"$2^{{{int(np.log2(v))}}}$")

    return {
        # "X": x_formatted,
        "X": x_values,
        "Y": scaled_components
    }


def plot_trends(var: str,log2=False, data=None, ylabel="Time(s)", **kwargs):
    export_fname = f"trends_{var}"
    if not data:
        df = get_var_df(var)
        data = {
            "X": df[var],
            "Y": [
            df["t_graph"],
            df["t_lineage"],
            df["t_schedule"],
            df["t_vrf"],
        ]
        }
    
    if log2:
        # data = log_scale_stacked_data(data)
        kwargs["log_y_ticks"]=True
        # ylabel = r"Time (s), log$_2$ scale"
    draw_hybrid(data,
                # component_labels = ["toposort", "rxshape", "stageprep", "verification"],
                component_labels = ["create SSA DAGs", "infer lineages", "determine stages", "verification"],
                ylabel="time(s)",
                export_fname=export_fname, show=True, legend=False, **kwargs)
    

# [markdown]
#  Plot

# ("nm", 3),
# ("tp", 2),
# ("dp", 2),
# ("layers", 2),
# ("pp", 1),
# ("gbs", 1),
# ("heads", 2),
# ("hidden", 2),
# ("seqlen", 1),
plot_trends("num_mb", highlight_idxs=[0], log2=True,  uniform_spacing=True, ylabel="")
plot_trends("num_tp", highlight_idxs=[], log2=True,  uniform_spacing=True, ylabel="")
plot_trends("num_dp", highlight_idxs=[], log2=True,  uniform_spacing=True, ylabel="")
plot_trends("num_layers", highlight_idxs=[], log2=True,  uniform_spacing=True, ylabel="")
plot_trends("num_pp", highlight_idxs=[], log2=True,  uniform_spacing=True, )
plot_trends("gbs", highlight_idxs=[], log2=True, uniform_spacing=True, ylabel="")
plot_trends("num_heads", highlight_idxs=[], log2=True, uniform_spacing=True, ylabel="")
plot_trends("hidden_size", highlight_idxs=[], log2=True, uniform_spacing=True, ylabel="")
plot_trends("seqlen", highlight_idxs=[], log2=True, uniform_spacing=True, ylabel="")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Bar components (stacked colors)
default_colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
labels = ["create SSA DAGs", "infer lineages", "determine stages", "verification"]

# Line legend (e.g., total)
line_color = "black"
line_label = "Total"

# Legend handles
handles = [Patch(color=color, label=label) for color, label in zip(default_colors, labels)]
handles.append(Line2D([0], [0], color=line_color, linewidth=1.5, label=line_label, marker='o'))

# Create standalone legend figure
fig_legend = plt.figure(figsize=(6, 0.6))
ax = fig_legend.add_subplot(111)
ax.axis('off')

# Draw legend
legend = ax.legend(
    handles=handles,
    ncol=5,
    loc='center',
    frameon=False
)

plt.tight_layout()
plt.savefig("figs/trends_legend.pdf", format="pdf", transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()











