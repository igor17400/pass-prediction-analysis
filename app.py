# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "altair",
# ]
# ///
import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("# Pitch Nets\n\nGradient boosting vs deep learning on open football event data. Models training, page coming.")
    return


if __name__ == "__main__":
    app.run()
