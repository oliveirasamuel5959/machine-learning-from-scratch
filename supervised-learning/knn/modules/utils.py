# ------------------------
# KNN Regression Metrics
# ------------------------
import pandas as pd

def save_metrics(mse, mae, r2, mape, outdir):
  metrics_df = pd.DataFrame({
      "MSE": [mse],
      "MAE": [mae],
      "R²": [r2],
      "MAPE (%)": [mape]
  })
  
  metrics_file = outdir / "knn_regression_metrics.json"
  metrics_df.to_json(metrics_file, orient="records", indent=4)
  print(f"\n[OK] Metrics saved to: {metrics_file}")