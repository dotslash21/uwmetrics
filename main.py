import pickle

from uw_metrics import UwMetrics

if __name__ == "__main__":
    uw_metrics = UwMetrics()
    metrics = uw_metrics.calculate()

    with open("data/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
