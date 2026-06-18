"""Generate a realistic sample dataset and 100 evaluation questions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
DATASET_PATH = DATA_DIR / "dataset.csv"
QUESTIONS_PATH = DATA_DIR / "evaluation_questions.json"
RNG = np.random.default_rng(547)


MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]


def build_dataset(n_rows: int = 420) -> pd.DataFrame:
    """Create a synthetic dataset with the Online Shoppers Intention schema."""
    rows = []
    month_weights = np.array([0.08, 0.08, 0.09, 0.09, 0.08, 0.11, 0.09, 0.12, 0.17, 0.09])

    for _ in range(n_rows):
        month = RNG.choice(MONTHS, p=month_weights)
        visitor = RNG.choice(VISITOR_TYPES, p=[0.68, 0.27, 0.05])
        weekend = bool(RNG.random() < 0.24)

        administrative = int(RNG.poisson(3))
        informational = int(RNG.poisson(1))
        product_related = int(max(1, RNG.poisson(28)))

        admin_duration = round(max(0.0, administrative * RNG.normal(42, 15)), 3)
        info_duration = round(max(0.0, informational * RNG.normal(30, 12)), 3)
        product_duration = round(max(0.0, product_related * RNG.normal(36, 11)), 3)

        bounce = float(np.clip(RNG.normal(0.045, 0.03), 0.0, 0.2))
        exit_rate = float(np.clip(bounce + RNG.normal(0.02, 0.02), 0.01, 0.25))
        special_day = float(np.clip(RNG.choice([0.0, 0.2, 0.4, 0.6, 0.8], p=[0.72, 0.1, 0.08, 0.06, 0.04]), 0.0, 1.0))

        operating_system = int(RNG.integers(1, 9))
        browser = int(RNG.integers(1, 14))
        region = int(RNG.integers(1, 10))
        traffic_type = int(RNG.choice(np.arange(1, 9), p=[0.11, 0.24, 0.17, 0.12, 0.1, 0.09, 0.09, 0.08]))

        month_boost = {"Nov": 0.16, "Oct": 0.06, "Dec": 0.04, "Aug": 0.03}.get(month, 0.0)
        visitor_boost = {"Returning_Visitor": 0.09, "New_Visitor": -0.015, "Other": -0.03}[visitor]
        weekend_boost = 0.02 if weekend else 0.0
        traffic_boost = {2: 0.08, 3: 0.05, 4: 0.03}.get(traffic_type, -0.01 if traffic_type == 8 else 0.0)
        page_values = max(0.0, RNG.normal(4.5, 7.0) + month_boost * 55 + visitor_boost * 35 + traffic_boost * 25)

        score = (
            -1.5
            + month_boost * 4
            + visitor_boost * 5
            + weekend_boost * 2
            + traffic_boost * 3
            + min(product_related / 60, 1.3)
            + min(page_values / 25, 2.0)
            - bounce * 6
            - exit_rate * 3
            - special_day * 0.4
        )
        probability = 1 / (1 + np.exp(-score))
        revenue = bool(RNG.random() < probability)

        if revenue:
            page_values = round(page_values + RNG.normal(8, 6), 3)
        else:
            page_values = round(max(0.0, page_values - RNG.normal(3, 2)), 3)

        rows.append(
            {
                "Administrative": administrative,
                "Administrative_Duration": round(admin_duration, 3),
                "Informational": informational,
                "Informational_Duration": round(info_duration, 3),
                "ProductRelated": product_related,
                "ProductRelated_Duration": round(product_duration, 3),
                "BounceRates": round(bounce, 4),
                "ExitRates": round(exit_rate, 4),
                "PageValues": page_values,
                "SpecialDay": special_day,
                "Month": month,
                "OperatingSystems": operating_system,
                "Browser": browser,
                "Region": region,
                "TrafficType": traffic_type,
                "VisitorType": visitor,
                "Weekend": weekend,
                "Revenue": revenue,
            }
        )

    return pd.DataFrame(rows)


def build_questions(df: pd.DataFrame) -> list[dict]:
    """Build 100 business questions with light ground truth from the data."""
    conversion_by_month = df.groupby("Month")["Revenue"].mean().sort_values(ascending=False)
    conversion_by_visitor = df.groupby("VisitorType")["Revenue"].mean().sort_values(ascending=False)
    conversion_by_traffic = df.groupby("TrafficType")["Revenue"].mean().sort_values(ascending=False)
    weekend_rate = df.groupby("Weekend")["Revenue"].mean()
    top_region = df.groupby("Region")["Revenue"].mean().sort_values(ascending=False).index[0]
    best_browser = df.groupby("Browser")["Revenue"].mean().sort_values(ascending=False).index[0]

    question_templates = [
        (
            "Which month has the highest conversion rate?",
            "trend",
            ["Month", "Revenue"],
            "categorical",
            f"{conversion_by_month.index[0]} has the highest conversion rate.",
            float(conversion_by_month.iloc[0]),
            ["month", "conversion", conversion_by_month.index[0].lower()],
        ),
        (
            "Which visitor type converts best?",
            "segmentation",
            ["VisitorType", "Revenue"],
            "categorical",
            f"{conversion_by_visitor.index[0]} converts best.",
            float(conversion_by_visitor.iloc[0]),
            ["visitor", "conversion", str(conversion_by_visitor.index[0]).lower()],
        ),
        (
            "Which traffic source drives the highest conversion?",
            "segmentation",
            ["TrafficType", "Revenue"],
            "categorical",
            f"TrafficType {conversion_by_traffic.index[0]} drives the highest conversion.",
            float(conversion_by_traffic.iloc[0]),
            ["traffic", "source", "conversion"],
        ),
        (
            "Do weekends perform better than weekdays?",
            "comparison",
            ["Weekend", "Revenue"],
            "directional",
            "Weekend sessions convert slightly better than weekday sessions."
            if weekend_rate[True] >= weekend_rate[False]
            else "Weekday sessions convert slightly better than weekend sessions.",
            float(weekend_rate.max()),
            ["weekend", "weekday", "conversion"],
        ),
        (
            "What actions would you recommend to improve conversion?",
            "recommendation",
            ["Revenue", "PageValues", "BounceRates", "VisitorType"],
            "actionable",
            "Reduce bounce rates, improve high-value product pages, and prioritize returning visitors.",
            None,
            ["recommendation", "bounce", "page value", "returning visitors"],
        ),
        (
            "How does bounce rate relate to conversion?",
            "correlation",
            ["BounceRates", "Revenue"],
            "correlation",
            "Higher bounce rates are associated with lower conversion.",
            None,
            ["bounce", "conversion", "negative"],
        ),
        (
            "How does page value relate to revenue?",
            "correlation",
            ["PageValues", "Revenue"],
            "correlation",
            "Higher page values are associated with higher revenue probability.",
            None,
            ["page value", "revenue", "positive"],
        ),
        (
            "Which region should marketing prioritize?",
            "recommendation",
            ["Region", "Revenue"],
            "categorical",
            f"Region {top_region} currently shows the strongest conversion signal.",
            float(top_region),
            ["region", "marketing", "prioritize"],
        ),
        (
            "Which browser segment shows the best conversion performance?",
            "segmentation",
            ["Browser", "Revenue"],
            "categorical",
            f"Browser {best_browser} shows the strongest conversion performance.",
            float(best_browser),
            ["browser", "conversion", "segment"],
        ),
        (
            "What patterns stand out among revenue-generating sessions?",
            "pattern",
            ["Revenue", "PageValues", "ProductRelated", "VisitorType"],
            "descriptive",
            "Revenue sessions tend to have higher page values, more product-related activity, and more returning visitors.",
            None,
            ["page values", "product", "returning visitors"],
        ),
    ]

    questions: list[dict] = []
    for i in range(100):
        template = question_templates[i % len(question_templates)]
        question, category, expected_variables, expected_type, ground_truth, numeric_answer, keywords = template
        questions.append(
            {
                "id": i + 1,
                "category": category,
                "question": question,
                "expected_variables": expected_variables,
                "expected_type": expected_type,
                "ground_truth": ground_truth,
                "numeric_answer": numeric_answer,
                "expected_keywords": keywords,
            }
        )
    return questions


def main() -> None:
    dataset = build_dataset()
    questions = build_questions(dataset)
    dataset.to_csv(DATASET_PATH, index=False)
    QUESTIONS_PATH.write_text(json.dumps(questions, indent=2))
    print(f"Created {DATASET_PATH}")
    print(f"Created {QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
