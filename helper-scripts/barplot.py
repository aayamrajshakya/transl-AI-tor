# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

import matplotlib.pyplot as plt
import numpy as np

metrics = ('BLEU', 'chrF') # x-axis labels
model_scores = {
    'Base': (19.6, 53.61),
    'Fine-tuned': (20.01, 51.01)
}  # score of each model

x = np.arange(len(metrics))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in model_scores.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=0)
    multiplier += 1


ax.set_ylabel('Score (%)', fontsize=12)
ax.set_ylim(0, 100) 
ax.set_title('Evaluation Scores by Model', fontsize=14)
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
ax.set_xticks(x + width/2, metrics)
ax.legend(loc='upper left', ncols=3)
ax.set_axisbelow(True) 
plt.tight_layout()
plt.savefig('score_graph.png', dpi=300)

plt.show()
