import csv
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_and_plot_results(file_path):
    # Initialize dictionaries to store results
    bins = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Read the CSV file
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reasoning_tokens = int(row['reasoning_tokens'])
            is_correct = row['is_correct'].lower() == 'true'
            
            # Determine which bin this row belongs to
            bin_key = (reasoning_tokens // 100) * 100
            
            bins[bin_key]['total'] += 1
            if is_correct:
                bins[bin_key]['correct'] += 1

    # Calculate accuracy for each bin
    token_bins = []
    accuracies = []
    sizes = []
    for token, results in sorted(bins.items()):
        accuracy = (results['correct'] / results['total']) * 100 if results['total'] > 0 else 0
        token_bins.append(f"{token}-{token+99}")
        accuracies.append(accuracy)
        sizes.append(results['total'] * 2)  # Multiply by 2 to make points more visible

    # Create the plot
    plt.figure(figsize=(14, 7))
    plt.scatter(token_bins, accuracies, s=sizes, alpha=0.6)
    plt.plot(token_bins, accuracies, '--', alpha=0.3)  # Add a light dashed line to connect points

    # Customize the plot
    plt.title('Accuracy vs. Number of Reasoning Tokens (100-token bins)')
    plt.xlabel('Number of Reasoning Tokens')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)  # Set y-axis to go from 0 to 105%

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for accuracy and sample size
    for i, (bin_range, accuracy, size) in enumerate(zip(token_bins, accuracies, sizes)):
        plt.annotate(f'{accuracy:.1f}% (n={size//2})', (i, accuracy), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Print tabular results
    print("\nBin Range | Accuracy | Correct / Total")
    print("-" * 40)
    for bin_range, accuracy, size in zip(token_bins, accuracies, sizes):
        correct = int(accuracy * size / 200)  # Calculate correct based on accuracy and total
        total = size // 2
        print(f"{bin_range:9} | {accuracy:7.2f}% | {correct:3d} / {total:3d}")

# Run the analysis and create the plot
analyze_and_plot_results('results.csv')
