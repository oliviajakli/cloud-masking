import matplotlib.pyplot as plt
from pathlib import Path

def save_figure(fig, path: Path, dpi=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # Close plot to prevent overlap.