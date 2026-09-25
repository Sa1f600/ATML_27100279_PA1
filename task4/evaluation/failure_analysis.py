"""save actual false acceptances for later visual and semantic interpretation."""

import numpy as np
import matplotlib.pyplot as plt

from task4.data.cifar10 import CLASSES
from task4.utils import save_csv, save_json


def failures(base, outputs, scores, threshold, output):
    rows, selected = [], []
    for group in ('near', 'far'):
        data, values = outputs[group], scores[group]
        candidates = np.flatnonzero(values <= threshold)
        candidates = candidates[np.argsort(values[candidates])]
        group_rows, seen = [], set()
        for index in candidates:
            row = {'group': group, 'image_index': int(data['indices'][index]),
                   'unknown_class': base.classes[int(data['labels'][index])],
                   'predicted_known_class': CLASSES[int(data['logits'][index, :10].argmax())],
                   'score': float(values[index]), 'threshold': float(threshold),
                   'semantic_interpretation': ''}
            rows.append(row)
            if row['unknown_class'] not in seen and len(group_rows) < 3:
                group_rows.append(row)
                seen.add(row['unknown_class'])
        # fill any remaining slots without inventing failures or changing the threshold
        for row in rows:
            if row['group'] == group and row not in group_rows and len(group_rows) < 3:
                group_rows.append(row)
        selected.extend(group_rows)
    save_csv(output / 'all_false_acceptances.csv', rows)
    save_csv(output / 'selected_failures.csv', selected)
    save_json(output / 'failure_counts.json', {
        group: {'false_acceptances': sum(r['group'] == group for r in rows),
                'shown': sum(r['group'] == group for r in selected)} for group in ('near', 'far')})
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row_axes, group in zip(axes, ('near', 'far')):
        group_rows = [r for r in selected if r['group'] == group]
        for index, axis in enumerate(row_axes):
            axis.axis('off')
            if index >= len(group_rows):
                axis.set_title(f'{group}: no further false acceptance')
                continue
            row = group_rows[index]
            axis.imshow(base[row['image_index']][0])
            axis.set_title(f"{group}: {row['unknown_class']} → {row['predicted_known_class']}\n"
                           f"MLS={row['score']:.3f}; threshold={threshold:.3f}", fontsize=10)
    figure.tight_layout()
    figure.savefig(output / 'failure_examples.png', dpi=180)
    plt.close(figure)
