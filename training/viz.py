import altair as alt
import pandas as pd

from typing import List, Optional, Union
from pathlib import Path


def plot_losses(
    epochs: List[int],
    train_losses: List[float],
    dev_losses: List[float],
    save_path: Optional[Union[Path, str]] = None,
    title: Optional[str] = None,
) -> None:

    assert (len(epochs) == len(dev_losses)) and (
        len(dev_losses) == len(train_losses)
    ), f"All sequences must be of the same length {len(epochs)=},{len(train_losses)=},{len(dev_losses)=}"

    source = pd.DataFrame(
        {
            "Epoch": epochs,
            "Train Loss": train_losses,
            "Dev Loss": dev_losses,
        }
    )
    df = source.melt(id_vars="Epoch", var_name="Set", value_name="Loss")
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Epoch:Q"),
            y=alt.Y("Loss:Q"),
            color=alt.Color(
                "Set:N",
                legend=alt.Legend(
                    title=title,
                ),
            ),
        )
    )

    if save_path is not None:
        chart.save(save_path)

    return chart
