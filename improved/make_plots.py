import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("improved/plots", exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ============================================================
# SHARED SETTINGS FOR ALL RUNS:
#   - Optimizer: AdamW
#   - LR schedule: linear warmup (10% of steps) + linear decay to 0
#   - Gradient clipping: max_norm=1.0
#   - Weight decay: 0.01 (no decay on bias/LayerNorm)
#   - LoRA runs: r=8, alpha=16 (unless noted)
# ============================================================

# ============================================================
# DATA — SST-5 + CFIMDB Classifier
# All runs: 20 epochs, batch_size=32
# ============================================================

# Full fine-tuning, lr=2e-5
sst_full_2e5 = [0.254, 0.381, 0.463, 0.474, 0.496, 0.492, 0.486, 0.485, 0.500, 0.501,
                0.500, 0.504, 0.505, 0.502, 0.506, 0.504, 0.507, 0.504, 0.504, 0.501]
# Full fine-tuning, lr=5e-5
sst_full_5e5 = [0.421, 0.476, 0.496, 0.514, 0.495, 0.503, 0.499, 0.482, 0.490, 0.481,
                0.473, 0.470, 0.478, 0.472, 0.475, 0.488, 0.490, 0.503, 0.496, 0.494]
# LoRA r=8 alpha=16, lr=1e-4 (full 20 epochs)
sst_lora_1e4 = [0.359, 0.489, 0.480, 0.488, 0.519, 0.490, 0.502, 0.494, 0.480, 0.472,
                0.484, 0.488, 0.484, 0.472, 0.472, 0.478, 0.479, 0.470, 0.470, 0.470]
# LoRA r=8 alpha=16, lr=3e-4 (epochs 5-19 captured)
sst_lora_3e4 = [np.nan]*5 + [0.508, 0.512, 0.523, 0.508, 0.511, 0.504, 0.508, 0.511, 0.517, 0.512,
                              0.509, 0.515, 0.516, 0.518, 0.515]
# LoRA r=8 alpha=16, lr=5e-4 (epochs 5-19 captured)
sst_lora_5e4 = [np.nan]*5 + [0.514, 0.519, 0.519, 0.521, 0.515, 0.500, 0.510, 0.510, 0.501, 0.498,
                              0.503, 0.501, 0.508, 0.500, 0.503]

# ============================================================
# DATA — Paraphrase Detection
# Full runs: batch_size=16, unless noted
# LoRA runs: batch_size=16, r=8, alpha=16
# ============================================================

# Full, lr=1e-5, 5 epochs, bs=16
para_full_1e5 = [0.845, 0.872, 0.884, 0.885, 0.886]
# Full, lr=2e-5, 5 epochs, bs=16
para_full_2e5 = [0.852, 0.877, 0.891, 0.893, 0.894]
# Full, lr=4e-4, 10 epochs, bs=64 (teammate's run)
para_full_4e4 = [0.877, 0.888, 0.891, 0.895, 0.891, 0.895, 0.899, 0.896, 0.894, 0.899]
# LoRA r=8 alpha=16, lr=1e-4, 5 epochs, bs=16
para_lora_1e4 = [0.828, 0.837, 0.851, 0.859, 0.859]
# LoRA r=8 alpha=16, lr=3e-4, 5 epochs, bs=16
para_lora_3e4 = [0.845, 0.854, 0.863, 0.873, 0.875]
# LoRA r=8 alpha=16, lr=5e-4, 5 epochs, bs=16
para_lora_5e4 = [0.848, 0.849, 0.865, 0.877, 0.880]

# ============================================================
# DATA — Sonnet Generation
# All runs: 10 epochs, batch_size=8
# Train loss per epoch (full fine-tuning only — LoRA only has final CHRF)
# ============================================================

# Full, lr=1e-5, 10 epochs, bs=8
sonnet_full_1e5_loss = [5.076, 4.590, 4.382, 4.262, 4.184, 4.145, 4.112, 4.091, 4.084]
# Full, lr=5e-5, 10 epochs, bs=8
sonnet_full_5e5_loss = [4.852, 4.260, 4.006, 3.836, 3.718, 3.650, 3.601, 3.555, 3.544, 3.524]

# Final Dev CHRF scores (LR sweep)
sonnet_chrf = {
    'Full\nlr=1e-5': 39.0,
    'Full\nlr=5e-5': 42.2,
    'LoRA\nlr=1e-4': 36.9,
    'LoRA\nlr=3e-4': 40.5,
    'LoRA\nlr=5e-4': 41.3,
}

# Alpha ablation: LoRA lr=1e-4, r=8, varying alpha
# SST: alpha=8 -> 0.501, alpha=16 -> 0.501, alpha=32 -> 0.501 (no effect)
# Sonnet:
sonnet_alpha_chrf = {'α=8': 37.1, 'α=16': 36.9, 'α=32': 40.2}

epochs_20 = list(range(20))
epochs_5 = list(range(5))

# ============================================================
# PLOT 1: SST-5 Dev Accuracy — All Configs
# Settings: 20 epochs, bs=32. Full: lr={2e-5, 5e-5}. LoRA: r=8, α=16, lr={1e-4, 3e-4, 5e-4}.
# Alpha ablation (α=8,16,32 at lr=1e-4): identical results (0.501), not plotted.
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs_20, sst_full_2e5, 'o-', label='Full, lr=2e-5 (best 0.507)', color='#2196F3', markersize=4)
ax.plot(epochs_20, sst_full_5e5, 's-', label='Full, lr=5e-5 (best 0.514)', color='#4CAF50', markersize=4)
ax.plot(epochs_20, sst_lora_1e4, '^--', label='LoRA, lr=1e-4 (best 0.519)', color='#FF9800', markersize=4)
ax.plot(epochs_20, sst_lora_3e4, 'v--', label='LoRA, lr=3e-4 (best 0.523)', color='#E91E63', markersize=4)
ax.plot(epochs_20, sst_lora_5e4, 'D--', label='LoRA, lr=5e-4 (best 0.521)', color='#9C27B0', markersize=4)

ax.set_xlabel('Epoch')
ax.set_ylabel('Dev Accuracy')
ax.set_title('SST-5 Sentiment Classification — Dev Accuracy\n(20 epochs, bs=32; LoRA: r=8, α=16)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.20, 0.56)
ax.text(0.02, 0.02, 'LoRA α ablation (α=8,16,32 at lr=1e-4): identical results (best=0.501 for all)',
        transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))
plt.tight_layout()
plt.savefig('improved/plots/sst_all_curves.png', bbox_inches='tight')
plt.close()
print("Saved sst_all_curves.png")

# ============================================================
# PLOT 2: Paraphrase Dev Accuracy — All Configs
# Full: bs=16, 5 epochs, lr={1e-5, 2e-5}; teammate: bs=64, 10 epochs, lr=4e-4.
# LoRA: r=8, α=16, bs=16, 5 epochs, lr={1e-4, 3e-4, 5e-4}.
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs_5, para_full_1e5, 'o-', label='Full, lr=1e-5, bs=16, 5ep (best 0.886)', color='#2196F3', markersize=6)
ax.plot(epochs_5, para_full_2e5, 's-', label='Full, lr=2e-5, bs=16, 5ep (best 0.894)', color='#4CAF50', markersize=6)
ax.plot(range(10), para_full_4e4, 'D-', label='Full, lr=4e-4, bs=64, 10ep (best 0.899)', color='#E91E63', markersize=6)
ax.plot(epochs_5, para_lora_1e4, '^--', label='LoRA, lr=1e-4, bs=16, 5ep (best 0.859)', color='#FF9800', markersize=6)
ax.plot(epochs_5, para_lora_3e4, 'v--', label='LoRA, lr=3e-4, bs=16, 5ep (best 0.875)', color='#9C27B0', markersize=6)
ax.plot(epochs_5, para_lora_5e4, 'P--', label='LoRA, lr=5e-4, bs=16, 5ep (best 0.880)', color='#795548', markersize=6)

ax.set_xlabel('Epoch')
ax.set_ylabel('Dev Accuracy')
ax.set_title('Paraphrase Detection — Dev Accuracy\n(LoRA: r=8, α=16)')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.80, 0.92)
plt.tight_layout()
plt.savefig('improved/plots/paraphrase_all_curves.png', bbox_inches='tight')
plt.close()
print("Saved paraphrase_all_curves.png")

# ============================================================
# PLOT 3: Sonnet Generation — Train Loss + LR CHRF + Alpha CHRF
# All runs: 10 epochs, bs=8. LoRA: r=8.
# ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Left: train loss curves (full fine-tuning only)
ax1.plot(range(len(sonnet_full_1e5_loss)), sonnet_full_1e5_loss, 'o-', label='Full, lr=1e-5', color='#2196F3', markersize=5)
ax1.plot(range(len(sonnet_full_5e5_loss)), sonnet_full_5e5_loss, 's-', label='Full, lr=5e-5', color='#4CAF50', markersize=5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss')
ax1.set_title('Training Loss\n(Full fine-tuning, 10ep, bs=8)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Middle: LR sweep CHRF bar chart
configs = list(sonnet_chrf.keys())
values = list(sonnet_chrf.values())
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
bars = ax2.bar(range(len(configs)), values, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels(configs, fontsize=9)
ax2.set_ylabel('Dev CHRF Score')
ax2.set_title('LR Sweep — Final Dev CHRF\n(10ep, bs=8; LoRA: r=8, α=16)')
ax2.set_ylim(30, 46)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Right: Alpha ablation CHRF bar chart
alpha_configs = list(sonnet_alpha_chrf.keys())
alpha_values = list(sonnet_alpha_chrf.values())
alpha_colors = ['#FF9800', '#FF7043', '#F44336']
bars = ax3.bar(range(len(alpha_configs)), alpha_values, color=alpha_colors, width=0.5, edgecolor='black', linewidth=0.5)
ax3.set_xticks(range(len(alpha_configs)))
ax3.set_xticklabels(alpha_configs, fontsize=10)
ax3.set_ylabel('Dev CHRF Score')
ax3.set_title('LoRA α Ablation\n(lr=1e-4, r=8, 10ep, bs=8)')
ax3.set_ylim(30, 46)
for bar, val in zip(bars, alpha_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

plt.suptitle('Sonnet Generation', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('improved/plots/sonnet_all_curves.png', bbox_inches='tight')
plt.close()
print("Saved sonnet_all_curves.png")

# ============================================================
# PLOT 4: Summary — Baseline vs Best Full vs Best LoRA
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

categories = ['Baseline', 'Best Full', 'Best LoRA']
bar_colors = ['#9E9E9E', '#2196F3', '#FF9800']

# SST-5
ax = axes[0]
vals = [0.493, 0.514, 0.523]
bars = ax.bar(categories, vals, color=bar_colors, width=0.5, edgecolor='black', linewidth=0.5)
ax.set_title('SST-5 Dev Accuracy')
ax.set_ylim(0.40, 0.56)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Paraphrase
ax = axes[1]
vals = [0.892, 0.899, 0.880]
bars = ax.bar(categories, vals, color=bar_colors, width=0.5, edgecolor='black', linewidth=0.5)
ax.set_title('Paraphrase Dev Accuracy')
ax.set_ylim(0.85, 0.92)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Sonnet
ax = axes[2]
vals = [39.0, 42.2, 41.3]
bars = ax.bar(categories, vals, color=bar_colors, width=0.5, edgecolor='black', linewidth=0.5)
ax.set_title('Sonnet Dev CHRF')
ax.set_ylim(30, 48)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}',
            ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Summary: Baseline vs Best Full Fine-tuning vs Best LoRA', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('improved/plots/summary_comparison.png', bbox_inches='tight')
plt.close()
print("Saved summary_comparison.png")

print("\nAll 4 plots saved to improved/plots/")
