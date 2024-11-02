import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict


def load_data_from_csv(filename):
    data = defaultdict(lambda: {'reasoning_tokens': [], 'latency': [], 'is_correct': []})
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            max_tokens = int(row['max_tokens'])
            data[max_tokens]['reasoning_tokens'].append(int(row['reasoning_tokens']))
            data[max_tokens]['latency'].append(float(row['latency']))
            data[max_tokens]['is_correct'].append(row['is_correct'].lower() == 'true')

    processed_data = []
    for max_tokens, values in sorted(data.items()):
        processed_data.append({
            'max_tokens': max_tokens,
            'avg_reasoning_tokens': np.mean(values['reasoning_tokens']),
            'avg_latency': np.mean(values['latency']),
            'accuracy': np.mean(values['is_correct']),
            'reasoning_tps': np.sum(values['reasoning_tokens']) / np.sum(values['latency'])
        })

    print(processed_data)
    return processed_data

def plot_results(data):
    max_completion_tokens = [d['max_tokens'] for d in data]
    avg_reasoning_tokens = [d['avg_reasoning_tokens'] for d in data]
    avg_latency = [d['avg_latency'] for d in data]
    accuracy = [d['accuracy'] for d in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot accuracy and average reasoning tokens
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(max_completion_tokens, accuracy, color='tab:blue', marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Avg Reasoning Tokens', color='tab:orange')
    ax1_twin.bar(max_completion_tokens, avg_reasoning_tokens, alpha=0.3, color='tab:orange', label='Avg Reasoning Tokens')
    ax1_twin.tick_params(axis='y', labelcolor='tab:orange')

    # Combine legends for the first subplot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot average latency
    ax2.set_xlabel('Max Completion Tokens')
    ax2.set_ylabel('Avg Reasoning Tokens Per Second', color='tab:green')
    ax2.plot(max_completion_tokens, avg_latency, color='tab:green', marker='^', label='Avg Latency')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper left')

    # Fit a line between accuracy and average reasoning tokens
    slope, intercept, r_value, p_value, std_err = stats.linregress(avg_reasoning_tokens, accuracy)
    line = slope * np.array(avg_reasoning_tokens) + intercept
    ax1.plot(max_completion_tokens, line, color='red', linestyle='--', label=f'Fit Line (R²={r_value**2:.2f})')

    plt.title(
        'Accuracy, Avg Reasoning Tokens, and Avg Latency vs Max Completion Tokens\n'
        'prompt: \"what\'s larger? 9.11 or 9.8? answer only from 9.11 or 9.8. please think step by step\"'
    )
    plt.tight_layout()
    plt.savefig('reasoning_analysis_plot.png')
    plt.close()

def main():
    csv_filename = 'results.csv'
    data = load_data_from_csv(csv_filename)
    plot_results(data)
    print("Plot saved as 'reasoning_analysis_plot.png'")

if __name__ == "__main__":
    main()
