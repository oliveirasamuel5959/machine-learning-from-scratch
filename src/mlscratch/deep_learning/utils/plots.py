import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# Plot and save history and plots
# ----------------------------------
def save_history_and_plots(history, output_dir, prefix=None):
  """

  Args:
      history (_type_): objeto retornado pelo model.fit
      output_dir (_type_): save directory
      prefix (_type_): ex.: "train" ou "val"
  """

  os.makedirs(output_dir, exist_ok=True)

  # ---------------------
  # History to Dataframe
  # ---------------------
  hist_df = pd.DataFrame(history)
  hist_df["epoch"] = hist_df.index + 1

  csv_path = os.path.join(output_dir, f"history.csv")
  hist_df.to_csv(csv_path, index=False)

  # ---------------------
  # Accuracy plot
  # ---------------------
  fig_acc, ax_acc = plt.subplots(figsize=(7, 5))

  ax_acc.plot(hist_df["epoch"], hist_df["train_acc"], label="Train Accuracy")

  if "val_acc" in hist_df:
    ax_acc.plot(hist_df["epoch"], hist_df["val_acc"], label="Validation Accuracy")

  ax_acc.set_xlabel("Epoch")
  ax_acc.set_ylabel("Accuracy")
  ax_acc.set_title("Training and Validation Accuracy")
  ax_acc.legend()
  ax_acc.grid(True)

  acc_plot_path = os.path.join(output_dir, f"acc_plot.png")
  fig_acc.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close(fig_acc)

  # ---------------------
  # Loss plot
  # ---------------------
  fig_loss, ax_loss = plt.subplots(figsize=(7, 5))

  ax_loss.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")

  if "val_loss" in hist_df:
    ax_loss.plot(hist_df["epoch"], hist_df["val_loss"], label="Validation Loss")

  ax_loss.set_xlabel("Epoch")
  ax_loss.set_ylabel("Loss")
  ax_loss.set_title("Training and Validation Loss")
  ax_loss.legend()
  ax_loss.grid(True)

  loss_plot_path = os.path.join(output_dir, f"loss_plot.png")
  fig_loss.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close(fig_loss)

  print()
  print(f"[OK] History saved to {csv_path}")
  print(f"[OK] Accuracy plot saved to {acc_plot_path}")
  print(f"[OK] Loss plot saved to {loss_plot_path}")
  
  
# ------------------------------------
# Plot and save confusion matrix
# ------------------------------------
def plot_and_save_confusion_matrix(cm, output_dir):
  fig, ax = plt.subplots(figsize=(15, 15))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

  ax.set_title("Confusion Matrix")
  ax.set_xlabel("Predicted labels")
  ax.set_ylabel("True labels")

  # ax.set_xticks(np.arange(len(labels)) + 0.5)
  # ax.set_yticks(np.arange(len(labels)) + 0.5)

  # ax.set_xticklabels(labels, rotation=45, ha='right')
  # ax.set_yticklabels(labels)

  # ax.set_ylim(len(labels), 0)

  plt.tight_layout()

  img_path = os.path.join(
    output_dir,
    f"confusion_matrix.png"
  )

  plt.savefig(img_path, dpi=300, bbox_inches="tight")
  plt.show()
  plt.close()
  
  print(f"[OK] Confusion matrix saved to {img_path}")