"""Posterior convergence report for the Model AL workflow."""

from pathlib import Path

from Scripts.aabc_convergence import (
    analyze_posterior_convergence,
    write_secondary_rss_diagnostic,
)


def main() -> None:
    for folder in sorted(Path(".").glob("Widx_*")):
        if not folder.is_dir():
            continue
        result = analyze_posterior_convergence(folder, model_label="Model AL")
        write_secondary_rss_diagnostic(folder)
        print(f"{folder}: converged={result['converged']}")


if __name__ == "__main__":
    main()
