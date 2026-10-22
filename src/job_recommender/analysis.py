from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


def _top_counter_frame(counter: Counter, top_n: int, label: str) -> pd.DataFrame:
    data = counter.most_common(top_n)
    return pd.DataFrame(data, columns=[label, "count"])


def _save_bar_chart(
    frame: pd.DataFrame,
    label: str,
    title: str,
    output_path: Path,
    total_count: int | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    chart = frame.iloc[::-1]
    bars = ax.barh(chart[label], chart["count"], color="#1f77b4")

    if total_count:
        labels = [
            f"{count:,.0f} ({count / total_count:.1%})"
            for count in chart["count"]
        ]
        ax.set_xlabel("Listings (count and share of dataset)")
    else:
        labels = [f"{count:,.0f}" for count in chart["count"]]
        ax.set_xlabel("Listings")

    ax.bar_label(bars, labels=labels, padding=4, fontsize=8)
    ax.set_xlim(0, chart["count"].max() * 1.2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_dataset(
    csv_path: str | Path,
    output_dir: str | Path = "outputs",
    chunksize: int = 100_000,
    top_n: int = 10,
) -> dict:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "Job Title",
        "Role",
        "Work Type",
        "Qualifications",
        "Country",
        "Job Description",
        "skills",
    ]

    row_count = 0
    title_counter: Counter[str] = Counter()
    work_type_counter: Counter[str] = Counter()
    qualification_counter: Counter[str] = Counter()
    country_counter: Counter[str] = Counter()
    missing_counter: Counter[str] = Counter()
    description_char_total = 0
    skill_char_total = 0

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        row_count += len(chunk)
        description_char_total += chunk["Job Description"].fillna("").str.len().sum()
        skill_char_total += chunk["skills"].fillna("").str.len().sum()

        for column in usecols:
            missing_counter[column] += int(chunk[column].isna().sum())

        title_counter.update(chunk["Job Title"].fillna("Unknown").tolist())
        work_type_counter.update(chunk["Work Type"].fillna("Unknown").tolist())
        qualification_counter.update(chunk["Qualifications"].fillna("Unknown").tolist())
        country_counter.update(chunk["Country"].fillna("Unknown").tolist())

    summary = {
        "row_count": row_count,
        "unique_job_titles": len(title_counter),
        "avg_job_description_chars": round(description_char_total / row_count, 2),
        "avg_skills_chars": round(skill_char_total / row_count, 2),
        "missing_counts": dict(missing_counter),
        "top_job_titles": dict(title_counter.most_common(top_n)),
        "top_work_types": dict(work_type_counter.most_common(top_n)),
        "top_qualifications": dict(qualification_counter.most_common(top_n)),
        "top_countries": dict(country_counter.most_common(top_n)),
    }

    job_titles_frame = _top_counter_frame(title_counter, top_n, "Job Title")
    work_types_frame = _top_counter_frame(work_type_counter, top_n, "Work Type")

    _save_bar_chart(
        job_titles_frame,
        label="Job Title",
        title="Most Common Job Titles",
        output_path=output_dir / "top_job_titles.png",
    )
    _save_bar_chart(
        work_types_frame,
        label="Work Type",
        title="Work Type Distribution",
        output_path=output_dir / "top_work_types.png",
        total_count=row_count,
    )

    summary_lines = [
        "# Job Recommendation Dataset Summary",
        "",
        f"- Total rows: {summary['row_count']:,}",
        f"- Unique job titles: {summary['unique_job_titles']}",
        f"- Average job description length: {summary['avg_job_description_chars']} characters",
        f"- Average skills length: {summary['avg_skills_chars']} characters",
        "",
        "## Missing Values",
    ]

    for column, value in summary["missing_counts"].items():
        summary_lines.append(f"- {column}: {value}")

    summary_lines.extend(["", "## Top Job Titles"])
    for name, value in summary["top_job_titles"].items():
        summary_lines.append(f"- {name}: {value:,}")

    summary_lines.extend(["", "## Top Work Types"])
    for name, value in summary["top_work_types"].items():
        summary_lines.append(f"- {name}: {value:,}")

    summary_lines.extend(["", "## Top Qualifications"])
    for name, value in summary["top_qualifications"].items():
        summary_lines.append(f"- {name}: {value:,}")

    summary_lines.extend(["", "## Top Countries"])
    for name, value in summary["top_countries"].items():
        summary_lines.append(f"- {name}: {value:,}")

    (output_dir / "dataset_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return summary
