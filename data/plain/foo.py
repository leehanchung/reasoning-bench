import csv
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_plot_results(file_path, num_buckets=10):
    # Read the CSV file
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reasoning_tokens = int(row['reasoning_tokens'])
            is_correct = row['is_correct'].lower() == 'true'
            data.append((reasoning_tokens, is_correct))

    # Sort data by reasoning tokens
    data.sort(key=lambda x: x[0])

    # Create equal-size buckets
    bucket_size = len(data) // num_buckets
    buckets = [data[i:i + bucket_size] for i in range(0, len(data), bucket_size)]
    
    # If there are any remaining data points, add them to the last bucket
    if len(buckets) > num_buckets:
        buckets[-2].extend(buckets[-1])
        buckets.pop()

    # Calculate accuracy for each bucket
    token_ranges = []
    accuracies = []
    sizes = []

    for bucket in buckets:
        min_token = min(item[0] for item in bucket)
        max_token = max(item[0] for item in bucket)
        correct = sum(1 for _, is_correct in bucket if is_correct)
        total = len(bucket)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        token_ranges.append(f"{min_token}-{max_token}")
        accuracies.append(accuracy)
        sizes.append(total * 5)  # Multiply by 5 to make points more visible

    # Create the plot
    plt.figure(figsize=(14, 7))
    plt.scatter(token_ranges, accuracies, s=sizes, alpha=0.6)
    plt.plot(token_ranges, accuracies, '--', alpha=0.3)  # Add a light dashed line to connect points

    # Customize the plot
    plt.title('Accuracy vs. Number of Reasoning Tokens (Equal-size buckets)')
    plt.xlabel('Number of Reasoning Tokens')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)  # Set y-axis to go from 0 to 105%

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for accuracy and sample size
    for i, (token_range, accuracy, size) in enumerate(zip(token_ranges, accuracies, sizes)):
        plt.annotate(f'{accuracy:.1f}% (n={size//5})', (i, accuracy), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Print tabular results
    print("\nToken Range | Accuracy | Correct / Total")
    print("-" * 45)
    for token_range, accuracy, size in zip(token_ranges, accuracies, sizes):
        correct = int(accuracy * size / 500)  # Calculate correct based on accuracy and total
        total = size // 5
        print(f"{token_range:11} | {accuracy:7.2f}% | {correct:3d} / {total:3d}")

# Run the analysis and create the plot
analyze_and_plot_results('results.csv', num_buckets=10)
