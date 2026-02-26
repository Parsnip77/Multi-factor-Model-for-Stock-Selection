"""
lgbm_model.py
-------------
LightGBM training engine for multi-factor alpha synthesis.

Wraps a single LGBMRegressor with finance-specific hyperparameters designed
to combat overfitting on noisy return data, plus utilities for feature
importance analysis and SHAP value visualisation.

Public API
----------
    model = AlphaLGBM()
    model.train(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)

    imp_df  = model.get_feature_importance(importance_type='gain')
    fig_imp = model.plot_feature_importance(fold=1, save_path=Path("plots/fi.png"))
    fig_shap = model.plot_shap(X_sample, save_path=Path("plots/shap.png"))

Notes
-----
SHAP analysis requires the ``shap`` library.  If it is not installed,
``plot_shap`` raises ImportError with an installation hint.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AlphaLGBM:
    """LightGBM regressor tuned for quantitative factor-synthesis tasks.

    Hyperparameter choices follow standard quant-ML practice:
    - ``objective='regression_l1'`` (MAE / L1 loss): robust to extreme
      return outliers that are common in financial data.
    - Shallow trees (``max_depth=4``, ``num_leaves=15``): prevents the model
      from memorising noise in low signal-to-noise financial series.
    - Column / row sub-sampling (``colsample_bytree``, ``subsample``):
      similar to a random-forest ensemble effect, reduces variance.
    - Early stopping on a held-out validation set: the single most effective
      guard against overfitting.
    """

    def __init__(self) -> None:
        self.model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Core training / inference
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Fit the model with early stopping monitored on the validation set.

        Parameters
        ----------
        X_train, X_val : pd.DataFrame
            Feature matrices; columns must match.
        y_train, y_val : pd.Series
            Target forward returns aligned to the respective feature rows.
        """
        self.feature_names = X_train.columns.tolist()

        callbacks = [
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=callbacks,
        )

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return predicted scores using the best iteration found by early stopping."""
        return self.model.predict(X_test)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(
        self, importance_type: str = "gain"
    ) -> pd.DataFrame:
        """Return a DataFrame of feature importances sorted descending.

        Parameters
        ----------
        importance_type : str
            ``'gain'`` (default) or ``'split'``.

        Returns
        -------
        pd.DataFrame
            Columns: ``['feature', 'importance']``, sorted by importance desc.
        """
        if importance_type == "gain":
            raw = self.model.booster_.feature_importance(importance_type="gain")
        else:
            raw = self.model.feature_importances_

        df_imp = pd.DataFrame(
            {"feature": self.feature_names, "importance": raw}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp

    def plot_feature_importance(
        self,
        fold: Optional[int] = None,
        save_path: Optional[pathlib.Path] = None,
    ) -> plt.Figure:
        """Bar chart of feature importances for this fold.

        Parameters
        ----------
        fold : int, optional
            Fold number shown in the title.
        save_path : Path, optional
            If provided, the figure is saved to this path (PNG).

        Returns
        -------
        matplotlib Figure
        """
        df_imp = self.get_feature_importance()

        fig, ax = plt.subplots(figsize=(8, max(4, len(df_imp) * 0.45)))
        ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
        title = "Feature Importance (gain)"
        if fold is not None:
            title += f" — Fold {fold}"
        ax.set_title(title)
        ax.set_xlabel("Importance (gain)")
        fig.tight_layout()

        if save_path is not None:
            save_path = pathlib.Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)

        return fig

    # ------------------------------------------------------------------
    # SHAP analysis
    # ------------------------------------------------------------------

    def plot_shap(
        self,
        X_sample: pd.DataFrame,
        save_path: Optional[pathlib.Path] = None,
        max_display: int = 15,
    ) -> plt.Figure:
        """SHAP beeswarm plot showing feature contribution direction & magnitude.

        Parameters
        ----------
        X_sample : pd.DataFrame
            A sample of the feature matrix used for SHAP computation.
            For performance, pass at most ~500 rows (sub-sample if needed).
        save_path : Path, optional
            If provided, the figure is saved to this path (PNG).
        max_display : int
            Number of top features to display (default 15).

        Returns
        -------
        matplotlib Figure

        Raises
        ------
        ImportError
            If the ``shap`` library is not installed.
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "The 'shap' library is required for SHAP analysis. "
                "Install it with: pip install shap"
            ) from exc

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        fig = plt.gcf()
        fig.suptitle("SHAP Feature Contribution (Beeswarm)", fontsize=13)
        fig.tight_layout()

        if save_path is not None:
            save_path = pathlib.Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
