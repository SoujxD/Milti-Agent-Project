"""Presentation generator agent focused on reusable ecommerce dataset insights."""

from __future__ import annotations

import math
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from utils.dataset_adapter import load_analysis_dataset


@dataclass(slots=True)
class SlideContent:
    """Structured content for a slide."""

    title: str
    bullets: list[str]
    speaker_notes: str
    chart_key: str | None = None


class PresentationGeneratorAgent:
    """Generates dataset-focused charts and a reusable PowerPoint deck."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_dir = self.output_dir / "charts"
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.docs_asset_dir = self.output_dir.parent / "docs" / "assets"
        self.docs_asset_dir.mkdir(parents=True, exist_ok=True)

    def _normalized(self, name: str) -> str:
        return "".join(ch.lower() for ch in str(name) if ch.isalnum())

    def _find_column(self, dataset: pd.DataFrame, keywords: list[str], exclude: set[str] | None = None) -> str | None:
        exclude = exclude or set()
        normalized_map = {column: self._normalized(column) for column in dataset.columns}
        for keyword in keywords:
            norm_keyword = self._normalized(keyword)
            for column, norm_column in normalized_map.items():
                if column in exclude:
                    continue
                if norm_keyword in norm_column:
                    return column
        return None

    def _target_series(self, series: pd.Series) -> pd.Series | None:
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int)
        if pd.api.types.is_numeric_dtype(series):
            cleaned = series.dropna()
            unique_values = set(cleaned.unique().tolist())
            if unique_values and unique_values.issubset({0, 1}):
                return series.fillna(0).astype(int)
        if pd.api.types.is_object_dtype(series):
            lowered = series.astype(str).str.strip().str.lower()
            mapping = {
                "true": 1, "false": 0,
                "yes": 1, "no": 0,
                "y": 1, "n": 0,
                "converted": 1, "not_converted": 0,
                "purchase": 1, "no_purchase": 0,
                "purchased": 1, "not_purchased": 0,
            }
            unique_values = set(lowered.dropna().unique().tolist())
            if unique_values and unique_values.issubset(set(mapping)):
                return lowered.map(mapping).fillna(0).astype(int)
        return None

    def infer_schema(self, dataset: pd.DataFrame) -> dict[str, str | None]:
        """Infer common ecommerce columns from flexible schemas."""
        target_col = self._find_column(
            dataset,
            ["revenue", "converted", "conversion", "purchase", "purchased", "ordered", "order"],
        )
        if target_col is None:
            for column in dataset.columns:
                if self._target_series(dataset[column]) is not None:
                    target_col = column
                    break

        time_col = self._find_column(dataset, ["month", "date", "week", "season", "day", "period"])
        customer_col = self._find_column(
            dataset,
            ["visitor", "customer", "segment", "user_type", "member", "cohort", "device", "browser", "brand", "category"],
            exclude={time_col} if time_col else set(),
        )
        channel_col = self._find_column(
            dataset,
            ["traffic", "channel", "source", "campaign", "medium", "acquisition", "referrer", "category", "brand"],
            exclude={time_col, customer_col} - {None},
        )
        engagement_col = self._find_column(
            dataset,
            ["pagevalue", "page_value", "pagevalues", "basket", "cart", "duration", "session", "product", "engagement", "event_count", "avg_price", "price"],
            exclude={time_col, customer_col, channel_col} - {None},
        )
        friction_col = self._find_column(
            dataset,
            ["bounce", "exit", "drop", "abandon", "friction"],
            exclude={time_col, customer_col, channel_col, engagement_col} - {None},
        )

        numeric_candidates = [
            column for column in dataset.select_dtypes(include=["number", "bool"]).columns
            if column not in {target_col, time_col, customer_col, channel_col, engagement_col, friction_col}
        ]
        fallback_numeric = numeric_candidates[0] if numeric_candidates else None

        return {
            "target": target_col,
            "time": time_col,
            "customer": customer_col,
            "channel": channel_col,
            "engagement": engagement_col or fallback_numeric,
            "friction": friction_col,
        }

    def _target_metric(self, dataset: pd.DataFrame, schema: dict[str, str | None]) -> pd.Series | None:
        target_col = schema["target"]
        if not target_col:
            return None
        return self._target_series(dataset[target_col])

    def _series_by_dimension(
        self,
        dataset: pd.DataFrame,
        schema: dict[str, str | None],
        dimension: str | None,
        top_n: int = 8,
    ) -> pd.Series | None:
        if not dimension:
            return None
        working = dataset.copy()
        metric = self._target_metric(working, schema)
        if metric is not None:
            working["_metric_target"] = metric
            series = working.groupby(dimension)["_metric_target"].mean().sort_values(ascending=False).head(top_n)
            return series
        series = working[dimension].astype(str).value_counts().head(top_n)
        return series.sort_values(ascending=False)

    def _bucket_series(self, dataset: pd.DataFrame, schema: dict[str, str | None]) -> tuple[pd.Series | None, str]:
        numeric_col = schema["engagement"] or schema["friction"]
        if not numeric_col or numeric_col not in dataset.columns:
            return None, ""
        numeric_series = pd.to_numeric(dataset[numeric_col], errors="coerce").dropna()
        if numeric_series.empty:
            return None, ""

        working = dataset.loc[numeric_series.index].copy()
        metric = self._target_metric(working, schema)

        quantiles = np.unique(np.quantile(numeric_series, [0, 0.25, 0.5, 0.75, 1.0]))
        if len(quantiles) < 3:
            bins = np.linspace(float(numeric_series.min()), float(numeric_series.max()) + 1e-6, 4)
        else:
            bins = quantiles
        labels = [f"Q{i + 1}" for i in range(len(bins) - 1)]
        bucketed = pd.cut(pd.to_numeric(working[numeric_col], errors="coerce"), bins=bins, labels=labels, include_lowest=True, duplicates="drop")
        if metric is not None:
            working["_metric_target"] = metric
            series = working.groupby(bucketed, observed=False)["_metric_target"].mean()
            return series, numeric_col
        series = working.groupby(bucketed, observed=False).size()
        return series, numeric_col

    def _format_metric_label(self, schema: dict[str, str | None], has_target: bool) -> str:
        return "Conversion Rate" if has_target else "Share / Count"

    def summarize_dataset(self, dataset_path: str | Path) -> dict[str, object]:
        adapted = load_analysis_dataset(dataset_path)
        dataset = adapted.dataframe
        schema = self.infer_schema(dataset)
        target_metric = self._target_metric(dataset, schema)
        time_series = self._series_by_dimension(dataset, schema, schema["time"])
        customer_series = self._series_by_dimension(dataset, schema, schema["customer"])
        channel_series = self._series_by_dimension(dataset, schema, schema["channel"])
        bucket_series, bucket_col = self._bucket_series(dataset, schema)

        target_rate = float(target_metric.mean()) if target_metric is not None else None
        top_customer = str(customer_series.index[0]) if customer_series is not None and len(customer_series) else "Top segment unavailable"
        top_channel = str(channel_series.index[0]) if channel_series is not None and len(channel_series) else "Top channel unavailable"
        top_time = str(time_series.index[0]) if time_series is not None and len(time_series) else "Peak period unavailable"

        engagement_col = schema["engagement"]
        friction_col = schema["friction"]
        engagement_delta = None
        friction_delta = None
        if target_metric is not None:
            revenue_mask = target_metric == 1
            non_revenue_mask = target_metric == 0
            if engagement_col and engagement_col in dataset.columns:
                engagement_series = pd.to_numeric(dataset[engagement_col], errors="coerce")
                engagement_delta = (
                    float(engagement_series[revenue_mask].mean()) if revenue_mask.any() else math.nan,
                    float(engagement_series[non_revenue_mask].mean()) if non_revenue_mask.any() else math.nan,
                )
            if friction_col and friction_col in dataset.columns:
                friction_series = pd.to_numeric(dataset[friction_col], errors="coerce")
                friction_delta = (
                    float(friction_series[revenue_mask].mean()) if revenue_mask.any() else math.nan,
                    float(friction_series[non_revenue_mask].mean()) if non_revenue_mask.any() else math.nan,
                )

        return {
            "dataset": dataset,
            "schema": schema,
            "rows": adapted.raw_rows,
            "columns": len(dataset.columns),
            "analysis_rows": adapted.analysis_rows,
            "analysis_grain": adapted.analysis_grain,
            "source_format": adapted.source_format,
            "target_rate": target_rate,
            "top_time": top_time,
            "top_customer": top_customer,
            "top_channel": top_channel,
            "engagement_delta": engagement_delta,
            "friction_delta": friction_delta,
            "bucket_col": bucket_col,
        }

    def generate_charts(self, dataset_path: str | Path) -> dict[str, Path]:
        dataset = load_analysis_dataset(dataset_path).dataframe
        schema = self.infer_schema(dataset)
        has_target = self._target_metric(dataset, schema) is not None
        ylabel = self._format_metric_label(schema, has_target)

        chart_specs = {
            "time_performance": (self._series_by_dimension(dataset, schema, schema["time"]), "Performance by Time Period", "bar"),
            "customer_performance": (self._series_by_dimension(dataset, schema, schema["customer"]), "Performance by Customer Segment", "barh"),
            "channel_performance": (self._series_by_dimension(dataset, schema, schema["channel"]), "Performance by Acquisition Channel", "bar"),
        }
        bucket_series, bucket_col = self._bucket_series(dataset, schema)
        chart_specs["engagement_buckets"] = (
            bucket_series,
            f"Performance by {bucket_col} Bucket" if bucket_col else "Performance by Numeric Bucket",
            "line",
        )

        palette = {
            "primary": "#1E5F74",
            "secondary": "#4F8FBF",
            "accent": "#7FC8A9",
            "warm": "#E7A33E",
        }
        plt.style.use("seaborn-v0_8-whitegrid")

        chart_paths: dict[str, Path] = {}
        for index, (key, (series, title, chart_type)) in enumerate(chart_specs.items()):
            if series is None or len(series) == 0:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.4))
            color = list(palette.values())[index % len(palette)]
            if chart_type == "bar":
                series.plot(kind="bar", ax=ax, color=color)
                ax.set_xlabel("")
                ax.set_ylabel(ylabel)
            elif chart_type == "barh":
                series.sort_values().plot(kind="barh", ax=ax, color=color)
                ax.set_xlabel(ylabel)
                ax.set_ylabel("")
            else:
                series.plot(kind="line", ax=ax, marker="o", linewidth=3, color=color)
                ax.set_xlabel("")
                ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=14)
            fig.tight_layout()
            output_path = self.chart_dir / f"{key}.png"
            fig.savefig(output_path, dpi=180)
            plt.close(fig)
            shutil.copy2(output_path, self.docs_asset_dir / output_path.name)
            chart_paths[key] = output_path
        return chart_paths

    def build_slide_contents(self, dataset_path: str | Path) -> list[SlideContent]:
        summary = self.summarize_dataset(dataset_path)
        schema = summary["schema"]
        target_text = (
            f"Average conversion / purchase rate is {summary['target_rate']:.1%}."
            if summary["target_rate"] is not None
            else "The uploaded dataset does not expose a clean binary conversion field, so the story focuses on distributions and segment patterns."
        )

        engagement_text = "Higher-intent sessions show stronger engagement signals."
        if summary["engagement_delta"] and not any(math.isnan(value) for value in summary["engagement_delta"]):
            engagement_col = schema["engagement"] or "engagement metric"
            engagement_text = (
                f"{engagement_col} is higher for converting sessions "
                f"({summary['engagement_delta'][0]:.2f} vs {summary['engagement_delta'][1]:.2f})."
            )

        friction_text = "Lower-friction journeys are more likely to convert."
        if summary["friction_delta"] and not any(math.isnan(value) for value in summary["friction_delta"]):
            friction_col = schema["friction"] or "friction metric"
            friction_text = (
                f"{friction_col} is lower for converting sessions "
                f"({summary['friction_delta'][0]:.3f} vs {summary['friction_delta'][1]:.3f})."
            )

        return [
            SlideContent(
                title="E-Commerce Customer Behavior Insights",
                bullets=[
                    "This presentation summarizes the uploaded ecommerce dataset in a stakeholder-friendly format.",
                    (
                        f"The uploaded file contains {summary['rows']} rows and is analyzed at the "
                        f"{summary['analysis_grain']} level across {summary['analysis_rows']} records."
                    ),
                    target_text,
                ],
                speaker_notes="Introduce the dataset story and frame the deck as a business-ready summary of customer behavior patterns.",
            ),
            SlideContent(
                title="Dataset Snapshot",
                bullets=[
                    "The dataset has been automatically profiled to detect customer segments, channel information, and business outcomes.",
                    f"Peak performance currently appears in {summary['top_time']}.",
                    f"Most informative customer grouping detected: {schema['customer'] or 'not clearly available'}.",
                    f"Most informative acquisition grouping detected: {schema['channel'] or 'not clearly available'}.",
                ],
                speaker_notes="Explain how the uploaded dataset was interpreted and which fields drove the analysis.",
                chart_key="time_performance",
            ),
            SlideContent(
                title="Target Customers",
                bullets=[
                    f"The strongest customer segment in the uploaded data is {summary['top_customer']}.",
                    f"The strongest acquisition or channel grouping is {summary['top_channel']}.",
                    "Priority audiences are the segments that combine stronger conversion signals with stronger engagement behavior.",
                    "These segments are the best candidates for campaign prioritization and personalized targeting.",
                ],
                speaker_notes="Highlight which customers appear most valuable and where acquisition quality is strongest.",
                chart_key="customer_performance",
            ),
            SlideContent(
                title="Behavioral Patterns",
                bullets=[
                    engagement_text,
                    friction_text,
                    f"Bucket analysis based on {summary['bucket_col'] or 'the strongest numeric metric'} shows where commercial performance improves.",
                    "Behavior and customer quality are working together rather than independently.",
                ],
                speaker_notes="Use this slide to explain the behavioral signals that separate stronger sessions from weaker ones.",
                chart_key="engagement_buckets",
            ),
            SlideContent(
                title="Key Findings",
                bullets=[
                    f"Top time period: {summary['top_time']}.",
                    f"Top customer segment: {summary['top_customer']}.",
                    f"Top channel grouping: {summary['top_channel']}.",
                    "The dataset suggests that acquisition quality and on-site engagement both influence business outcomes.",
                ],
                speaker_notes="Condense the analysis into the most portable executive findings.",
                chart_key="channel_performance",
            ),
            SlideContent(
                title="Recommendations",
                bullets=[
                    "Focus budget on the strongest channels and customer groups identified in the analysis.",
                    "Improve weak customer journeys by reducing friction and strengthening discovery paths.",
                    "Use high-intent engagement signals to trigger tailored offers, retention flows, or remarketing.",
                    "Track the same dimensions over time to validate whether conversion improves after changes.",
                ],
                speaker_notes="Translate the patterns into actions across marketing, merchandising, and experience optimization.",
            ),
            SlideContent(
                title="Conclusion",
                bullets=[
                    "The uploaded ecommerce dataset can be turned into a clear commercial story without manual chart building.",
                    "Target segments, channel quality, and engagement behavior are the main drivers surfaced by the analysis.",
                    "This workflow is designed to help analysts move quickly from raw data to presentation-ready outputs.",
                ],
                speaker_notes="Close by emphasizing speed, clarity, and repeatability for future ecommerce datasets.",
            ),
        ]

    def _apply_theme(self, slide) -> None:
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor(251, 252, 250)

    def _add_title(self, slide, title: str) -> None:
        title_box = slide.shapes.add_textbox(Inches(0.55), Inches(0.35), Inches(8.8), Inches(0.65))
        paragraph = title_box.text_frame.paragraphs[0]
        run = paragraph.add_run()
        run.text = title
        run.font.size = Pt(24)
        run.font.bold = True
        run.font.color.rgb = RGBColor(24, 64, 84)

    def _add_bullets(self, slide, bullets: list[str]) -> None:
        text_box = slide.shapes.add_textbox(Inches(0.75), Inches(1.18), Inches(5.25), Inches(5.0))
        frame = text_box.text_frame
        frame.word_wrap = True
        for index, bullet in enumerate(bullets[:5]):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.text = bullet
            paragraph.level = 0
            paragraph.font.size = Pt(16)
            paragraph.space_after = Pt(7)

    def _add_notes(self, slide, speaker_notes: str) -> None:
        try:
            slide.notes_slide.notes_text_frame.text = speaker_notes
        except Exception:
            footer = slide.shapes.add_textbox(Inches(0.7), Inches(6.85), Inches(8.2), Inches(0.3))
            paragraph = footer.text_frame.paragraphs[0]
            paragraph.text = f"Speaker notes: {speaker_notes}"
            paragraph.alignment = PP_ALIGN.LEFT
            paragraph.font.size = Pt(9)
            paragraph.font.color.rgb = RGBColor(96, 112, 122)

    def _add_chart(self, slide, chart_path: Path) -> None:
        slide.shapes.add_picture(str(chart_path), Inches(6.05), Inches(1.32), width=Inches(3.18))

    def create_presentation(self, dataset_path: str | Path, output_path: str | Path) -> tuple[Path, list[SlideContent], dict[str, Path]]:
        chart_paths = self.generate_charts(dataset_path)
        slides = self.build_slide_contents(dataset_path)

        presentation = Presentation()
        presentation.slide_width = Inches(10)
        presentation.slide_height = Inches(7.5)
        layout = presentation.slide_layouts[6]

        for slide_content in slides:
            slide = presentation.slides.add_slide(layout)
            self._apply_theme(slide)
            self._add_title(slide, slide_content.title)
            wrapped = [textwrap.fill(item, width=57) for item in slide_content.bullets]
            self._add_bullets(slide, wrapped)
            if slide_content.chart_key and slide_content.chart_key in chart_paths:
                self._add_chart(slide, chart_paths[slide_content.chart_key])
            self._add_notes(slide, slide_content.speaker_notes)

        output_path = Path(output_path)
        presentation.save(output_path)
        shutil.copy2(output_path, self.docs_asset_dir / output_path.name)
        return output_path, slides, chart_paths
