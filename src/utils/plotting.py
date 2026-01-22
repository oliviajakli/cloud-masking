import matplotlib.pyplot as plt     # type: ignore
from matplotlib.figure import Figure    # type: ignore
from pathlib import Path

def save_figure(fig: Figure, path: Path, dpi: int = 300) -> None:
    """Save a matplotlib figure to the specified path.
    params:
        fig: matplotlib.figure.Figure, the figure to save
        path: Path, the file path to save the figure
        dpi: int, resolution in dots per inch
    returns: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # Close plot to prevent overlap.