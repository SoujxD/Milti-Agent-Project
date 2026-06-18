"""Deterministic EDA agent for generalized ecommerce datasets."""

from __future__ import annotations

import math
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.dataset_adapter import DatasetAdapterResult, load_analysis_dataset, read_dataset


@dataclass(slots=True)
class EDAChart:
    """Chart metadata plus file path."""

    key: str
    title: str
    caption: str
    path: Path


class EDAAgent:
    """Profiles datasets, runs quality checks, generates charts, and prepares analyst handoff context."""

    CACHE_VERSION = "light-eda-v2"
    CHART_DPI = 140

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.chart_dir = self.output_dir / "charts"
        self.docs_asset_dir = self.output_dir.parent / "docs" / "assets"
        self.cache_dir = self.output_dir / "eda_cache"

    def _ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.docs_asset_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _normalized(self, name: str) -> str:
        return "".join(ch.lower() for ch in str(name) if ch.isalnum())

    def _find_column(self, dataset: pd.DataFrame, keywords: list[str], exclude: set[str] | None = None) -> str | None:
        exclude = exclude or set()
        normalized_map = {column: self._normalized(column) for column in dataset.columns}
        for keyword in keywords:
            probe = self._normalized(keyword)
            for column, normalized in normalized_map.items():
                if column in exclude:
                    continue
                if probe in normalized:
                    return column
        return None

    def _target_series(self, series: pd.Series) -> pd.Series | None:
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int)
        if pd.api.types.is_numeric_dtype(series):
            cleaned = pd.to_numeric(series, errors="coerce").dropna()
            unique_values = set(cleaned.unique().tolist())
            if unique_values and unique_values.issubset({0, 1}):
                return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
        non_null = series.dropna()
        lowered = non_null.astype("string").str.strip().str.lower()
        mapping = {
            "1": 1, "0": 0,
            "true": 1, "false": 0, "yes": 1, "no": 0,
            "y": 1, "n": 0, "purchase": 1, "no_purchase": 0,
            "purchased": 1, "not_purchased": 0, "converted": 1, "not_converted": 0,
        }
        unique_values = set(lowered.dropna().unique().tolist())
        if unique_values and unique_values.issubset(set(mapping)):
            normalized = series.astype("string").str.strip().str.lower()
            return normalized.map(mapping).fillna(0).astype(int)
        return None

    def _is_identifierish(self, column: str, series: pd.Series) -> bool:
        normalized = self._normalized(column)
        if any(token in normalized for token in ["id", "uuid", "identifier"]):
            return True
        if not pd.api.types.is_numeric_dtype(series):
            return False
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty or numeric.nunique(dropna=True) != len(numeric):
            return False
        ordered = numeric.sort_values().reset_index(drop=True)
        steps = ordered.diff().dropna()
        return not steps.empty and steps.nunique() == 1

    def _resolve_target_column(self, dataset: pd.DataFrame) -> str | None:
        keyword_matches: list[str] = []
        normalized_map = {column: self._normalized(column) for column in dataset.columns}
        for keyword in ["revenue", "purchased", "converted", "conversion", "purchase_count", "purchase", "order"]:
            probe = self._normalized(keyword)
            keyword_matches.extend(
                column
                for column, normalized in normalized_map.items()
                if probe in normalized and column not in keyword_matches
            )
        for column in keyword_matches:
            if self._target_series(dataset[column]) is not None:
                return column
        for column in dataset.columns:
            if self._target_series(dataset[column]) is not None:
                return column
        return None

    def infer_schema(self, dataset: pd.DataFrame) -> dict[str, str | None]:
        """Infer generalized ecommerce roles from the analysis dataset."""
        target = self._resolve_target_column(dataset)

        time = self._find_column(dataset, ["event_month", "month", "date", "week", "day", "period", "session_start"])
        customer = self._find_column(
            dataset,
            ["visitor", "customer", "segment", "cohort", "member", "device", "browser", "brand"],
            exclude={time} - {None},
        )
        channel = self._find_column(
            dataset,
            ["traffic", "channel", "source", "campaign", "medium", "acquisition", "referrer", "category"],
            exclude={time, customer} - {None},
        )
        product = self._find_column(dataset, ["product", "sku", "item", "category", "brand"], exclude={customer, channel, time} - {None})
        if product and pd.api.types.is_numeric_dtype(dataset[product]) and dataset[product].nunique(dropna=True) > max(12, len(dataset) * 0.02):
            product = None
        engagement = self._find_column(
            dataset,
            ["cart_count", "event_count", "pagevalue", "page_value", "duration", "session", "product", "basket", "engagement", "avg_price"],
            exclude={time, customer, channel, product} - {None},
        )
        friction = self._find_column(dataset, ["bounce", "exit", "drop", "abandon", "friction"], exclude={time, customer, channel, product, engagement} - {None})

        return {
            "target": target,
            "time": time,
            "customer": customer,
            "channel": channel,
            "product": product,
            "engagement": engagement,
            "friction": friction,
        }

    def _target_metric(self, dataset: pd.DataFrame, schema: dict[str, str | None]) -> pd.Series | None:
        target_col = schema.get("target")
        if not target_col or target_col not in dataset.columns:
            return None
        return self._target_series(dataset[target_col])

    def _group_metric(self, dataset: pd.DataFrame, dimension: str | None, metric: pd.Series | None, top_n: int = 10) -> pd.Series | None:
        if not dimension or dimension not in dataset.columns:
            return None
        working = dataset.copy()
        if metric is not None:
            working["_target_metric"] = metric
            grouped = working.groupby(dimension)["_target_metric"].mean().sort_values(ascending=False).head(top_n)
            return grouped
        grouped = working[dimension].astype(str).value_counts().head(top_n)
        return grouped.sort_values(ascending=False)

    def _count_missing(self, dataset: pd.DataFrame) -> pd.Series:
        return (dataset.isna().mean() * 100).sort_values(ascending=False)

    def _numeric_columns(self, dataset: pd.DataFrame, schema: dict[str, str | None]) -> list[str]:
        excluded = {value for value in schema.values() if value}
        cols: list[str] = []
        for column in dataset.select_dtypes(include=["number", "bool"]).columns:
            if column not in excluded and not self._is_identifierish(column, dataset[column]):
                cols.append(column)
        return cols

    def _numeric_bucket_metric(self, dataset: pd.DataFrame, metric_col: str | None, target_metric: pd.Series | None) -> tuple[pd.Series | None, str]:
        if not metric_col or metric_col not in dataset.columns:
            return None, ""
        numeric = pd.to_numeric(dataset[metric_col], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            return None, metric_col
        quantiles = np.unique(np.quantile(valid, [0, 0.25, 0.5, 0.75, 1.0]))
        bins = quantiles if len(quantiles) >= 3 else np.linspace(float(valid.min()), float(valid.max()) + 1e-6, 5)
        labels = [f"Q{i + 1}" for i in range(len(bins) - 1)]
        bucketed = pd.cut(numeric, bins=bins, labels=labels, include_lowest=True, duplicates="drop")
        if target_metric is not None:
            working = pd.DataFrame({"bucket": bucketed, "target": target_metric})
            series = working.groupby("bucket", observed=False)["target"].mean()
            return series, metric_col
        counts = bucketed.value_counts().sort_index()
        return counts, metric_col

    def _top_outliers(self, dataset: pd.DataFrame) -> list[dict[str, Any]]:
        outliers: list[dict[str, Any]] = []
        for column in dataset.select_dtypes(include=["number"]).columns:
            series = pd.to_numeric(dataset[column], errors="coerce").dropna()
            if len(series) < 8:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if math.isclose(iqr, 0.0):
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((series < lower) | (series > upper)).sum())
            if count:
                outliers.append({"column": column, "count": count, "share": count / max(len(series), 1)})
        return sorted(outliers, key=lambda item: item["count"], reverse=True)[:6]

    def _quality_report(self, raw_dataset: pd.DataFrame, analysis_dataset: pd.DataFrame, schema: dict[str, str | None]) -> dict[str, Any]:
        duplicate_rows = int(raw_dataset.duplicated().sum())
        missing_share = self._count_missing(analysis_dataset)
        null_heavy_columns = [
            {"column": column, "missing_pct": round(float(value), 2)}
            for column, value in missing_share.items()
            if value > 20
        ][:10]
        constant_columns = [column for column in analysis_dataset.columns if analysis_dataset[column].nunique(dropna=False) <= 1][:10]
        high_cardinality = [
            {"column": column, "unique_values": int(analysis_dataset[column].nunique(dropna=True))}
            for column in analysis_dataset.select_dtypes(include=["object"]).columns
            if analysis_dataset[column].nunique(dropna=True) > max(25, len(analysis_dataset) * 0.05)
        ][:10]
        outliers = self._top_outliers(analysis_dataset)

        warnings: list[str] = []
        if duplicate_rows:
            warnings.append(f"{duplicate_rows:,} duplicate raw rows detected.")
        if not schema.get("target"):
            warnings.append("No clean target or outcome column was detected.")
        if null_heavy_columns:
            warnings.append(f"{len(null_heavy_columns)} columns have more than 20% missing values.")
        if constant_columns:
            warnings.append(f"{len(constant_columns)} columns are constant or nearly constant.")
        if not warnings:
            warnings.append("No major structural data quality issues were detected.")

        return {
            "duplicate_rows": duplicate_rows,
            "missing_summary": [
                {"column": column, "missing_pct": round(float(value), 2)}
                for column, value in missing_share.head(12).items()
            ],
            "null_heavy_columns": null_heavy_columns,
            "constant_columns": constant_columns,
            "high_cardinality_columns": high_cardinality,
            "numeric_outliers": outliers,
            "warnings": warnings,
        }

    def _profile(self, adapted: DatasetAdapterResult, schema: dict[str, str | None]) -> dict[str, Any]:
        dataset = adapted.dataframe
        target_metric = self._target_metric(dataset, schema)
        target_rate = float(target_metric.mean()) if target_metric is not None else None
        time_series = self._group_metric(dataset, schema.get("time"), target_metric)
        customer_series = self._group_metric(dataset, schema.get("customer"), target_metric)
        channel_series = self._group_metric(dataset, schema.get("channel"), target_metric)
        product_series = self._group_metric(dataset, schema.get("product"), target_metric)
        numeric_columns = self._numeric_columns(dataset, schema)

        return {
            "source_format": adapted.source_format,
            "analysis_grain": adapted.analysis_grain,
            "raw_rows": adapted.raw_rows,
            "analysis_rows": adapted.analysis_rows,
            "columns": len(dataset.columns),
            "schema": schema,
            "target_rate": target_rate,
            "top_time": str(time_series.index[0]) if time_series is not None and len(time_series) else "Unavailable",
            "top_customer": str(customer_series.index[0]) if customer_series is not None and len(customer_series) else "Unavailable",
            "top_channel": str(channel_series.index[0]) if channel_series is not None and len(channel_series) else "Unavailable",
            "top_product_dimension": str(product_series.index[0]) if product_series is not None and len(product_series) else "Unavailable",
            "numeric_columns": numeric_columns[:10],
        }

    def _suggested_questions(self, schema: dict[str, str | None], quality_report: dict[str, Any]) -> list[str]:
        questions: list[str] = []
        if schema.get("customer"):
            questions.append(f"Which {schema['customer']} values should we prioritize based on performance?")
        if schema.get("channel"):
            questions.append(f"Which {schema['channel']} groups appear strongest for future investment?")
        if schema.get("product"):
            questions.append(f"Which {schema['product']} values show the strongest commercial potential?")
        if schema.get("time"):
            questions.append(f"How does performance change across {schema['time']}?")
        if schema.get("engagement"):
            questions.append(f"How does {schema['engagement']} relate to purchase or conversion behavior?")
        if quality_report["warnings"]:
            questions.append("Which data quality issues should we address before deeper analysis?")
        return questions[:6]

    def _key_findings(self, dataset: pd.DataFrame, profile: dict[str, Any], schema: dict[str, str | None], quality_report: dict[str, Any]) -> list[str]:
        findings = []
        if profile["target_rate"] is not None:
            findings.append(f"Detected outcome rate is {profile['target_rate']:.1%} at the {profile['analysis_grain']} level.")
        if profile["top_customer"] != "Unavailable":
            findings.append(f"Strongest detected customer grouping is {profile['top_customer']}.")
        if profile["top_channel"] != "Unavailable":
            findings.append(f"Strongest detected channel or acquisition grouping is {profile['top_channel']}.")
        if profile["top_time"] != "Unavailable":
            findings.append(f"Best-performing time grouping is {profile['top_time']}.")
        if quality_report["null_heavy_columns"]:
            top_missing = quality_report["null_heavy_columns"][0]
            findings.append(f"Data completeness risk: {top_missing['column']} is {top_missing['missing_pct']:.1f}% missing.")
        numeric = schema.get("engagement")
        if numeric and numeric in dataset.columns:
            findings.append(f"{numeric} is the most useful detected engagement-style metric for this dataset.")
        return findings[:6]

    def _handoff_summary(
        self,
        dataset: pd.DataFrame,
        profile: dict[str, Any],
        quality_report: dict[str, Any],
        suggested_questions: list[str],
    ) -> dict[str, Any]:
        schema = profile["schema"]
        key_findings = self._key_findings(dataset, profile, schema, quality_report)
        return {
            "dataset_type": profile["source_format"],
            "analysis_grain": profile["analysis_grain"],
            "target_column": schema.get("target"),
            "time_column": schema.get("time"),
            "customer_dimensions": [value for value in [schema.get("customer")] if value],
            "channel_dimensions": [value for value in [schema.get("channel"), schema.get("product")] if value],
            "engagement_metric": schema.get("engagement"),
            "friction_metric": schema.get("friction"),
            "quality_warnings": quality_report["warnings"][:5],
            "top_patterns": key_findings[:4],
            "recommended_questions": suggested_questions[:5],
        }

    def _retrieval_chunks(self, profile: dict[str, Any], quality_report: dict[str, Any], suggested_questions: list[str], chart_manifest: list[dict[str, Any]]) -> list[str]:
        chunks = [
            f"EDA profile: source format {profile['source_format']}, analysis grain {profile['analysis_grain']}, raw rows {profile['raw_rows']}, analysis rows {profile['analysis_rows']}.",
            f"Schema detection: target={profile['schema'].get('target')}, time={profile['schema'].get('time')}, customer={profile['schema'].get('customer')}, channel={profile['schema'].get('channel')}, product={profile['schema'].get('product')}, engagement={profile['schema'].get('engagement')}.",
            f"Top findings: top customer={profile['top_customer']}, top channel={profile['top_channel']}, top time={profile['top_time']}, target rate={profile['target_rate']}.",
        ]
        chunks.extend(f"Quality warning: {warning}" for warning in quality_report["warnings"][:5])
        chunks.extend(f"Suggested analyst question: {question}" for question in suggested_questions[:5])
        chunks.extend(f"Chart insight: {chart['title']} — {chart['caption']}" for chart in chart_manifest[:4])
        return chunks

    def _plot_bar(self, series: pd.Series, title: str, path: Path, horizontal: bool = False, ylabel: str = "Value") -> None:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        color = "#1E5F74"
        if horizontal:
            series.sort_values().plot(kind="barh", ax=ax, color=color)
            ax.set_xlabel(ylabel)
            ax.set_ylabel("")
        else:
            series.plot(kind="bar", ax=ax, color=color)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            plt.setp(ax.get_xticklabels(), rotation=22, ha="right")
        ax.set_title(title, fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=self.CHART_DPI)
        plt.close(fig)

    def _plot_hist(self, series: pd.Series, title: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.hist(series.dropna(), bins=18, color="#4F8FBF", edgecolor="white", alpha=0.9)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(series.name or "Value")
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=self.CHART_DPI)
        plt.close(fig)

    def _plot_heatmap(self, corr: pd.DataFrame, title: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        img = ax.imshow(corr.values, cmap="GnBu", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=28, ha="right", fontsize=8)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_title(title, fontsize=13)
        for row in range(corr.shape[0]):
            for col in range(corr.shape[1]):
                value = corr.iloc[row, col]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color="#153847")
        fig.colorbar(img, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout()
        fig.savefig(path, dpi=self.CHART_DPI)
        plt.close(fig)

    def _plot_area(self, series: pd.Series, title: str, path: Path, ylabel: str = "Value") -> None:
        cleaned = pd.to_numeric(series, errors="coerce").dropna()
        if cleaned.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 4.4))
        x = np.arange(len(cleaned))
        y = cleaned.values.astype(float)
        ax.plot(x, y, color="#1E5F74", linewidth=2.2)
        ax.fill_between(x, y, color="#9ED8DB", alpha=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([str(value) for value in cleaned.index], rotation=22, ha="right")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=self.CHART_DPI)
        plt.close(fig)

    def _plot_doughnut(self, series: pd.Series, title: str, path: Path) -> None:
        cleaned = series.dropna()
        if cleaned.empty:
            return
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        palette = ["#1E5F74", "#4F8FBF", "#9ED8DB", "#FFC857", "#F29E4C", "#C7D3DD"]
        wedges, _ = ax.pie(
            cleaned.values,
            startangle=90,
            colors=palette[: len(cleaned)],
            wedgeprops={"width": 0.42, "edgecolor": "white"},
        )
        ax.legend(
            wedges,
            [str(value) for value in cleaned.index],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=9,
        )
        ax.set_title(title, fontsize=13)
        fig.tight_layout()
        fig.savefig(path, dpi=self.CHART_DPI, bbox_inches="tight")
        plt.close(fig)

    def _plot_scatter(self, dataset: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path) -> None:
        frame = dataset[[x_col, y_col]].copy()
        frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
        frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
        frame = frame.dropna().head(1500)
        if frame.empty:
            return
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        ax.scatter(frame[x_col], frame[y_col], alpha=0.45, s=24, color="#1E5F74", edgecolors="none")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=self.CHART_DPI)
        plt.close(fig)

    def _render_chart(self, path: Path, render: Any) -> bool:
        if path.exists():
            path.unlink()
        render()
        return path.exists()

    def _chart_manifest(
        self,
        dataset: pd.DataFrame,
        profile: dict[str, Any],
        quality_report: dict[str, Any],
        chart_prefix: str,
    ) -> list[dict[str, Any]]:
        self._ensure_dirs()
        schema = profile["schema"]
        target_metric = self._target_metric(dataset, schema)
        charts: list[EDAChart] = []
        max_charts = 4

        missing = pd.Series(
            {row["column"]: row["missing_pct"] for row in quality_report["missing_summary"] if row["missing_pct"] > 0}
        ).sort_values(ascending=False)
        if not missing.empty:
            path = self.chart_dir / f"{chart_prefix}_missingness.png"
            if self._render_chart(path, lambda: self._plot_bar(missing.head(10), "Missing Values by Column", path, horizontal=True, ylabel="Missing %")):
                charts.append(EDAChart("missingness", "Missing values by column", "Shows the columns with the highest percentage of missing values.", path))

        time_series = self._group_metric(dataset, schema.get("time"), target_metric)
        if time_series is not None and len(time_series) and len(charts) < max_charts:
            path = self.chart_dir / f"{chart_prefix}_time_area.png"
            if self._render_chart(
                path,
                lambda: self._plot_area(
                    time_series.head(12),
                    "Trend Across Time",
                    path,
                    ylabel="Outcome Rate" if target_metric is not None else "Volume",
                )
            ):
                charts.append(
                    EDAChart(
                        "time_area",
                        "Trend across time",
                        "Uses a light area chart to show how the main business signal changes across the detected time field.",
                        path,
                    )
                )

        mix_dimension = schema.get("channel") or schema.get("customer") or schema.get("product")
        if mix_dimension and mix_dimension in dataset.columns and len(charts) < max_charts:
            mix_series = dataset[mix_dimension].astype(str).value_counts().head(5)
            if len(mix_series):
                path = self.chart_dir / f"{chart_prefix}_mix_doughnut.png"
                if self._render_chart(path, lambda: self._plot_doughnut(mix_series, f"Share of Top {mix_dimension} Groups", path)):
                    charts.append(
                        EDAChart(
                            "mix_doughnut",
                            f"Share of top {mix_dimension} groups",
                            "Summarizes the dominant business segments with a compact doughnut view.",
                            path,
                        )
                    )

        primary_numeric = schema.get("engagement")
        if not primary_numeric and profile["numeric_columns"]:
            primary_numeric = profile["numeric_columns"][0]
        numeric_candidates = dataset.select_dtypes(include=["number"]).copy()
        if (
            primary_numeric
            and primary_numeric in dataset.columns
            and schema.get("target")
            and schema["target"] in dataset.columns
            and len(charts) < max_charts
        ):
            path = self.chart_dir / f"{chart_prefix}_engagement_scatter.png"
            if self._render_chart(
                path,
                lambda: self._plot_scatter(dataset, primary_numeric, schema["target"], f"{primary_numeric} vs {schema['target']}", path),
            ):
                charts.append(
                    EDAChart(
                        "engagement_scatter",
                        f"{primary_numeric} vs {schema['target']}",
                        "Shows how a strong numeric behavior signal relates to the detected outcome variable.",
                        path,
                    )
                )
        elif len(numeric_candidates.columns) >= 2 and len(charts) < max_charts:
            corr = numeric_candidates.corr(numeric_only=True).round(2)
            if corr.shape[0] > 1:
                path = self.chart_dir / f"{chart_prefix}_correlation_heatmap.png"
                if self._render_chart(path, lambda: self._plot_heatmap(corr.iloc[:6, :6], "Numeric Correlation Heatmap", path)):
                    charts.append(
                        EDAChart(
                            "correlation_heatmap",
                            "Numeric correlation heatmap",
                            "Highlights the strongest numeric relationships in the normalized analysis view.",
                            path,
                        )
                    )

        if len(charts) < 3:
            distribution_col = None
            for candidate in [schema.get("engagement"), *profile["numeric_columns"]]:
                if candidate and candidate in dataset.columns:
                    distribution_col = candidate
                    break
            if distribution_col:
                path = self.chart_dir / f"{chart_prefix}_distribution.png"
                if self._render_chart(
                    path,
                    lambda: self._plot_hist(pd.to_numeric(dataset[distribution_col], errors="coerce"), f"Distribution of {distribution_col}", path),
                ):
                    charts.append(
                        EDAChart(
                            "distribution",
                            f"Distribution of {distribution_col}",
                            "Shows how the most informative detected numeric field is distributed across the dataset.",
                            path,
                        )
                    )

        manifest: list[dict[str, Any]] = []
        for chart in charts[:max_charts]:
            shutil.copy2(chart.path, self.docs_asset_dir / chart.path.name)
            manifest.append({
                "key": chart.key,
                "title": chart.title,
                "caption": chart.caption,
                "path": str(chart.path),
                "filename": chart.path.name,
            })
        return manifest

    def _cache_path(self, cache_key: str) -> Path:
        self._ensure_dirs()
        return self.cache_dir / f"{self.CACHE_VERSION}_{cache_key}.json"

    def _load_cached_report(self, cache_key: str) -> dict[str, Any] | None:
        cache_path = self._cache_path(cache_key)
        if not cache_path.exists():
            return None
        try:
            cached = json.loads(cache_path.read_text())
        except Exception:
            return None
        for chart in cached.get("chart_manifest", []):
            filename = chart.get("filename")
            if not filename or not (self.chart_dir / filename).exists():
                return None
            chart["path"] = str(self.chart_dir / filename)
        return cached

    def _store_cached_report(self, cache_key: str, report: dict[str, Any]) -> None:
        cache_path = self._cache_path(cache_key)
        cacheable = json.loads(json.dumps(report, default=str))
        for chart in cacheable.get("chart_manifest", []):
            chart["path"] = chart.get("filename", chart.get("path"))
        cache_path.write_text(json.dumps(cacheable, indent=2))

    def analyze_dataset(
        self,
        dataset_path: str | Path,
        include_charts: bool = True,
        chart_prefix: str | None = None,
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        if cache_key:
            cached = self._load_cached_report(cache_key)
            if cached is not None:
                return cached
        adapted = load_analysis_dataset(dataset_path)
        raw_dataset = read_dataset(dataset_path)
        analysis_dataset = adapted.dataframe
        schema = self.infer_schema(analysis_dataset)
        profile = self._profile(adapted, schema)
        quality_report = self._quality_report(raw_dataset, analysis_dataset, schema)
        suggested_questions = self._suggested_questions(schema, quality_report)
        prefix = chart_prefix or uuid.uuid4().hex[:12]
        chart_manifest = self._chart_manifest(analysis_dataset, profile, quality_report, prefix) if include_charts else []
        key_findings = self._key_findings(analysis_dataset, profile, schema, quality_report)
        handoff_summary = self._handoff_summary(analysis_dataset, profile, quality_report, suggested_questions)
        retrieval_chunks = self._retrieval_chunks(profile, quality_report, suggested_questions, chart_manifest)
        report = {
            "profile": profile,
            "quality_checks": quality_report,
            "chart_manifest": chart_manifest,
            "suggested_questions": suggested_questions,
            "handoff_summary": handoff_summary,
            "retrieval_chunks": retrieval_chunks,
            "key_findings": key_findings,
            "preview": raw_dataset.head(12).fillna("").to_dict(orient="records"),
        }
        if cache_key:
            self._store_cached_report(cache_key, report)
        return report
