import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_gradient_directions(history, output_path):
    if not history:
        return

    steps = [row["step"] for row in history]
    cosines = [row["cosine"] for row in history]

    plt.figure(figsize=(10, 4))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(steps, cosines, linewidth=1.5)
    plt.scatter(steps, cosines, s=10)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("Training step")
    plt.ylabel("cos(grad walk_loss, grad link_loss)")
    plt.title("Gradient Direction Agreement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()