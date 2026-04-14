import matplotlib.pyplot as plt

models = ['base', 'fine-tuned'] # x-axis labels
scores = [19.6, 47.03]  # bleu score of each model
fig, ax = plt.subplots(figsize=(5, 5))
bars = ax.bar(models, scores, color='#f26d30', width=0.5)
ax.bar_label(bars, label_type='center', color='black', fontsize=14) # this displays the raw score inside the bar for better view
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_ylim(0, 100) 
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True) 
plt.tight_layout()
plt.savefig('bleu_bar_graph.png', dpi=300)
