import datetime
import pandas as pd
import tempfile
from mini_rec_sys.utils import get_date, get_time_now


class Logger:
    def __init__(self, log_dir=None) -> None:
        if log_dir:
            self.runname = f"run_{get_date()}_{get_time_now()}"
            self.filename = f"{log_dir}/{self.runname}.csv"
        else:
            _, path = tempfile.mkstemp(suffix=".csv", delete=True)
            self.filename = path
        self.stats = []
        self.init_time = datetime.datetime.now()

    def log(self, stats: dict):
        """Log statistics for the current iteration."""
        diff = (datetime.datetime.now() - self.init_time).total_seconds() / 60
        stats.update({"elapsed_mins": round(diff, 1)})
        self.stats.append(stats)
        df = pd.DataFrame(self.stats)
        df.to_csv(self.filename, index=False)

    def latest_is_best_model(self, col: str):
        """
        Check if the latest iteration is the best model based on the statistic
        in `col`.
        """
        if len(self.stats) == 0:
            return False

        latest = self.stats[-1].get(col, None)
        if latest is None:
            return False

        metric = pd.DataFrame(self.stats)[col].dropna()
        if len(metric) == 0:
            return True

        best = max(metric)
        return latest >= best

    def has_not_progressed(self, col, k=5):
        """
        Check whether the metric in col has not progressed in the last k rounds.
        """
        metrics = [row[col] for row in self.stats if row.get(col, None)]
        if len(metrics) <= k:
            return False
        latest = metrics[-k:]
        previous = metrics[:-k]
        return max(latest) <= max(previous)

