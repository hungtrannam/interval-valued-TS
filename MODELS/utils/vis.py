import optuna.visualization.matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import torch
from statsmodels.graphics.tsaplots import plot_acf


def set_plot_style():
    plt.rcParams.update({
        'font.size': 15,
        'font.family': 'serif',
        'mathtext.fontset': 'cm', 
        'axes.linewidth': 0.8,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'legend.frameon': False,
        'legend.fontsize': 13,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',     
    })

def plot_metrics(study, save_dir):
    set_plot_style()
    trials = study.trials_dataframe()

    best_trial_idx = trials['value'].idxmin()
    best_val = trials['value'].min()

    plt.figure(figsize=(8, 5))
    plt.plot(trials['value'], marker='o', label='Validation Loss (MSE)', color=cm.viridis(0.5))
    plt.scatter(best_trial_idx, best_val, color=cm.viridis(0.9), zorder=5, label=...)
    plt.axvline(x=best_trial_idx, color=cm.viridis(0.9), linestyle='--', alpha=0.6)


    plt.xlabel('Trial')
    plt.ylabel('Validation Loss (MSE)')
    plt.title('Validation Loss per Trial')
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'ValLoss.pdf'))
    plt.close()

    for metric in ['user_attrs_dtw', 'user_attrs_mae']:
        if metric in trials.columns:
            plt.figure()
            plt.plot(trials[metric], marker='o')
            plt.xlabel('Trial')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} per Trial')
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{metric.upper()}_trial.pdf'))
            plt.close()

def plot_hyperparameter_importance(study, save_dir):
    set_plot_style()
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(15, 6))
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.figure.savefig(os.path.join(save_dir, 'HyImpo.pdf'))
    print(f"Saved Hyperparameter Importance at {save_dir}/HyImpo.pdf")
    plt.tight_layout()
    plt.close()

def plot_input(dataset, save_path='./figs/input_scaled_plot.pdf'):
    file_path = os.path.join(dataset.root_path, dataset.data_path)
    df_raw = pd.read_csv(file_path)

    # Lấy danh sách các cột input scaler đã dùng
    if dataset.features == 'MS':
        input_cols = list(df_raw.columns)
        input_cols.remove('date')
    elif dataset.features == 'M':
        input_cols = list(df_raw.columns)
        input_cols.remove('date')
        input_cols.remove(dataset.target)
    elif dataset.features == 'S':
        input_cols = [dataset.target]

    df_input_all = df_raw[input_cols]
    data_scaled_input = dataset.scaler.transform(df_input_all.values)

    # Plot chỉ các cột cần
    plot_cols = ['CO2']
    df_input_plot = df_raw[plot_cols]
    df_target = df_raw[[dataset.target]]
    num_plot = len(plot_cols) + 1

    train_border = dataset.num_train
    val_border = train_border + dataset.num_vali
    test_border = val_border + dataset.num_test

    plt.figure(figsize=(20, 3 * num_plot))

    def draw_split_lines():
        y_top = plt.ylim()[1]
        plt.axvline(train_border, color='gray', linestyle='--')
        plt.axvline(val_border, color='gray', linestyle='--')
        plt.text(train_border / 2, y_top * 0.9, 'Train', ha='center', color='gray')
        plt.text((train_border + val_border) / 2, y_top * 0.9, 'Val', ha='center', color='gray')
        plt.text((val_border + test_border) / 2, y_top * 0.9, 'Test', ha='center', color='gray')

    for i, col in enumerate(plot_cols):
        col_idx = input_cols.index(col)

        plt.subplot(num_plot, 2, i * 2 + 1)
        plt.plot(df_input_plot[col], label='Original', linewidth=3)
        draw_split_lines()
        plt.title(f'Original: {col}')
        plt.grid()

        plt.subplot(num_plot, 2, i * 2 + 2)
        plt.plot(data_scaled_input[:, col_idx], label='Scaled', color='orange', linewidth=3)
        draw_split_lines()
        plt.title(f'Scaled: {col}')
        plt.grid()

    # Plot target
    plt.subplot(num_plot, 2, num_plot * 2 - 1)
    plt.plot(df_target[dataset.target], label='Original', color='green', linewidth=3)
    draw_split_lines()
    plt.title(f'Original: {dataset.target}')
    plt.grid()

    plt.subplot(num_plot, 2, num_plot * 2)
    target_idx = input_cols.index(dataset.target)
    plt.plot(data_scaled_input[:, target_idx], label='Scaled', color='green', linewidth=3)
    draw_split_lines()
    plt.title(f'Scaled: {dataset.target}')
    plt.grid()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_residual_acf(y_true, y_pred, save_path='./ResidualACF.pdf', lags=48):
    set_plot_style()
    residuals = y_true.flatten() - y_pred.flatten()
    plt.figure(figsize=(10,5))
    plot_acf(residuals, lags=lags)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ========================== USING =====================================


def visual_interval(true, preds=None, name='./pic/test_interval.pdf', input_len=None):
    """
    Vẽ khoảng (lower, upper) và prediction khoảng.
    """
    set_plot_style()
    plt.figure(figsize=(15, 6))

    # Tách lower và upper
    true_lower = true[:, 0]
    true_upper = true[:, 1]

    plt.plot(true_lower, label='GroundTruth Lower', color='blue', linewidth=2)
    plt.plot(true_upper, label='GroundTruth Upper', color='blue', linestyle='--', linewidth=2)
    plt.fill_between(range(len(true_lower)), true_lower, true_upper, color='blue', alpha=0.2)

    if preds is not None:
        pred_lower = preds[:, 0]
        pred_upper = preds[:, 1]

        plt.plot(pred_lower, label='Prediction Lower', color='orange', linewidth=2)
        plt.plot(pred_upper, label='Prediction Upper', color='orange', linestyle='--', linewidth=2)
        plt.fill_between(range(len(pred_lower)), pred_lower, pred_upper, color='orange', alpha=0.2)

    if input_len is not None and input_len < len(true_lower):
        plt.axvline(input_len - 1, color='gray', linestyle='--', alpha=0.7)

    plt.xlabel("Time Steps")
    plt.ylabel("Target Values")
    plt.title("GroundTruth vs Prediction (Interval)")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_loss(train_losses=None, val_losses=None, name='./pic/loss.pdf'):
    set_plot_style()
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.semilogy(epochs, train_losses, label='Train Loss (log)', color=cm.viridis(0.1), linestyle='--', linewidth=3)
    if val_losses is not None:
        plt.semilogy(epochs, val_losses, label='Validation Loss (log)', color=cm.viridis(0.8), linewidth=4)

    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    print(f"Saved loss curve at {name}")
    plt.close()


def plot_optimization_history(study, save_dir):
    set_plot_style()
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(15, 6))
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig.figure.savefig(os.path.join(save_dir, 'OptHistory.pdf'))
    print(f"Saved Hyperparameter Importance at {save_dir}/OptHistory.pdf")
    plt.tight_layout()
    plt.close()

def plot_scatter_truth_vs_pred(y_true, y_pred, save_path='./PredScatter.pdf'):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from utils.metrics import metric
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm

    set_plot_style()

    # Tính metrics tổng thể
    mae0, mse0, rmse0, mape0, mspe0, nse0 = metric(y_pred[..., 0], y_true[..., 0])
    mae1, mse1, rmse1, mape1, mspe1, nse1 =  metric(y_pred[..., 1], y_true[..., 1])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ['Low', 'High']
    components = [0, 1]

    for ax, comp_idx, title in zip(axes, components, titles):
        y_true_comp = y_true[..., comp_idx]
        y_pred_comp = y_pred[..., comp_idx]

        y_true_flat = y_true_comp.flatten().reshape(-1, 1)
        y_pred_flat = y_pred_comp.flatten().reshape(-1, 1)

        # Hồi quy tuyến tính
        reg = LinearRegression().fit(y_true_flat, y_pred_flat)
        y_fit = reg.predict(y_true_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)

        # Màu theo batch
        num_batches = y_true.shape[0]
        colors = cm.get_cmap('viridis', num_batches)

        for i in range(num_batches):
            ax.scatter(
                y_true_comp[i].flatten(),
                y_pred_comp[i].flatten(),
                alpha=0.5,
                color=colors(i),
                label=f'Batch {i+1}' if i < 10 else None  # tránh legend quá dài
            )

        min_val = min(y_true_comp.min(), y_pred_comp.min())
        max_val = max(y_true_comp.max(), y_pred_comp.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

        ax.plot(y_true_flat, y_fit, color='black', linewidth=2, label=f'Fit (R²={r2:.4f})')

        ax.set_xlabel('Observed')
        ax.set_ylabel('Forecasted')
        ax.set_title(f'{title} Values')
        ax.legend(loc='lower right', fontsize=8)

        # Ghi metric trên subplot "Low"
        if comp_idx == 0:
            ax.annotate(
                f'MAE:  {mae0:.4f}\n'
                f'MSE:  {mse0:.4f}\n'
                f'RMSE: {rmse0:.4f}\n'
                f'MAPE: {mape0:.2f}%\n'
                f'MSPE: {mspe0:.2f}%\n'
                f'NSE:  {nse0:.4f}\n',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                fontsize=10,
            )
                # Ghi metric trên subplot "Low"
        if comp_idx == 1:
            ax.annotate(
                f'MAE:  {mae1:.4f}\n'
                f'MSE:  {mse1:.4f}\n'
                f'RMSE: {rmse1:.4f}\n'
                f'MAPE: {mape1:.2f}%\n'
                f'MSPE: {mspe1:.2f}%\n'
                f'NSE:  {nse1:.4f}\n',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
