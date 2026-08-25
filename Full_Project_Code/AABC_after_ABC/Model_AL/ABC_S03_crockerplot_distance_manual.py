"""Manual wrapper for the versioned AABC Step 3 loss computation."""

from ABC_S03_crockerplot_distance_aabc_multiple_save_sep import main


MANUAL_BATCH_LABEL = 75_000


if __name__ == "__main__":
    main([str(MANUAL_BATCH_LABEL)])
