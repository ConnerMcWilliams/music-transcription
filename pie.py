import matplotlib.pyplot as plt

def split_number(n):
    # Calculate approximate splits
    a = round(n * 0.6)
    b = round(n * 0.2)
    c = n - a - b
    # Adjust to ensure all are integers and sum to n
    parts = [a, b, c]
    while sum(parts) != n:
        diff = n - sum(parts)
        # Adjust the largest part
        idx = parts.index(max(parts))
        parts[idx] += diff
    return parts

def plot_pie(n):
    values = split_number(n)
    labels = []
    total = sum(values)
    for v in values:
        percent = round(100 * v / total, 2)
        labels.append(f"{percent}% ({v})")
    plt.figure(figsize=(6,6))
    plt.pie(values, labels=labels, autopct=None, startangle=90)
    plt.show()

plot_pie(16393)