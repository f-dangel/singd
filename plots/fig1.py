import matplotlib.pyplot as plt
import tikzplotlib as tpl
import seaborn as sns
import numpy as np
import pandas as pd
import wandb
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--force_pull', default=False, action='store_true')
args = parser.parse_args()

METHODS = ['kfac32', 'kfac16', 'ikfac32', 'ikfac16', 'sngd32', 'sngd16']
METHOD2SWEEP = {
    'kfac32': 'ssngd/cnn-exp/gsj2o87i',
    'kfac16': 'ssngd/cnn-exp/ovoabvma',
    'ikfac32': 'ssngd/cnn-exp/pjhbpnzo',
    'ikfac16': 'ssngd/cnn-exp/4dnmyhsv',
    'sngd32': 'ssngd/cnn-exp/v0848gp0',
    'sngd16': 'ssngd/cnn-exp/vvg1in9z',
}
FP2LINESTYLE = {
    '32': 'solid',
    '16': 'dashed',
}
COLORS = sns.color_palette('colorblind')
METHOD2COLOR = {
    'kfac': COLORS[0],
    'ikfac': COLORS[1],
    'sngd': COLORS[2],
}
MARKERS = ['o', '*']
FP2MARKER = {
    '32': MARKERS[0],
    '16': MARKERS[1],
}


# Plot styling
# # ICML
# WIDTH = 6.75133
# NeurIPS & ICLR
WIDTH = 5.50107
HEIGHT = 9.00177
SCRIPTSIZE = 7
FOOTNOTESIZE = 9

params = {
    'text.usetex': False,
    'font.size': SCRIPTSIZE,
    'font.family': 'Times New Roman',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.5,
    'axes.titlesize': SCRIPTSIZE+1,
    'axes.labelsize': SCRIPTSIZE+1,
    'lines.linewidth': 1,
    'xtick.major.size': 1.5,
    'ytick.major.size': 1.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.fontsize': SCRIPTSIZE-1,
    'pdf.fonttype': 42,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,bm}',
}
plt.rcParams.update(params)

# PULL DATA

# For caching
dir_name = './cache'
fname = 'fig1_best_runs.npy'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

if not os.path.exists(f'{dir_name}/{fname}') or args.force_pull:
    api = wandb.Api()
    results = {}

    for method, sweep_id in METHOD2SWEEP.items():
        sweep = api.sweep(sweep_id)
        best_run = sweep.best_run()

        test_acc = best_run.history(samples=None)['acc1']
        time_elapsed = best_run.summary['_runtime']
        system_metrics = best_run.history(stream='events')
        gpu_mem = system_metrics['system.gpu.0.memoryAllocatedBytes']

        results[method] = {
            'test_acc': test_acc,
            'time_elapsed': time_elapsed / 60,  # Minutes
            'peak_gpu_mem': gpu_mem.max() / 1e9  # GB
        }

    np.save(f'{dir_name}/{fname}', results)
else:
    results = np.load(f'{dir_name}/{fname}', allow_pickle=True).item()

# PLOT
fig, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True)
fig.set_size_inches(1*WIDTH, 0.15*HEIGHT)

colors = []

for method in METHODS:
    # Left plot
    x = results[method]['test_acc'].index
    y = results[method]['test_acc']

    label = method[:-2].upper() if '32' in method else None
    color = METHOD2COLOR[method[:-2]]
    if label is not None:
        colors.append(color)

    axs[0].plot(
        x+1, y, alpha=1,
        label=label, c=color, ls=FP2LINESTYLE[method[-2:]]
    )

    # Right plot
    runtime = results[method]['time_elapsed']
    peak_mem = results[method]['peak_gpu_mem']
    axs[1].scatter(
        runtime, peak_mem,
        color=METHOD2COLOR[method[:-2]],
        marker=FP2MARKER[method[-2:]]
    )

axs[0].set_xlim(1, 121)
axs[0].set_xlabel('Epoch')
axs[0].set_yticks([0, 25, 50, 75])
axs[0].set_ylabel('Test acc. (\%)')
# axs[0].legend()

dummy_lines = []
for _, v in FP2LINESTYLE:
    dummy_lines.append(axs[0].plot([],[], c='black', ls=v)[0])
lines = axs[0].get_lines()
legend1 = plt.legend(lines, ['KFAC', 'IKFAC', 'SNGD'])
legend2 = plt.legend([dummy_lines[i] for i in [0,1]], ["b = 0.5", "b = 0.8"])
axs[0].add_artist(legend1)

axs[1].set_xlabel('Training time (min)')
axs[1].set_ylabel('Peak mem (GB)')

f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f('s', colors[i]) for i in range(3)]
handles += [f(list(FP2MARKER.values())[i], 'k') for i in range(2)]
labels = ['KFAC', 'IKFAC', 'SNGD', 'FP-32', 'FP-16']
axs[1].legend(handles, labels)

plt.savefig('figs/fig1.pdf')
