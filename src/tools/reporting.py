from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("ai-ml-template")

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  --bg: #1e1e2e; --surface: #282840; --border: #3a3a5c;
  --text: #cdd6f4; --text-muted: #a6adc8; --accent: #89b4fa;
  --green: #a6e3a1; --red: #f38ba8; --orange: #fab387;
}}
[data-theme="light"] {{
  --bg: #eff1f5; --surface: #ffffff; --border: #ccd0da;
  --text: #4c4f69; --text-muted: #6c6f85; --accent: #1e66f5;
  --green: #40a02b; --red: #d20f39; --orange: #fe640b;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.6;
  padding: 24px;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 2rem; margin-bottom: 8px; color: var(--accent); }}
.subtitle {{ color: var(--text-muted); font-size: 0.9rem; margin-bottom: 32px; }}
.theme-toggle {{
  position: fixed; top: 16px; right: 16px;
  background: var(--surface); border: 1px solid var(--border);
  color: var(--text); padding: 8px 16px; border-radius: 8px;
  cursor: pointer; font-size: 0.9rem; z-index: 1000;
}}
.theme-toggle:hover {{ border-color: var(--accent); }}
.section {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; margin-bottom: 24px; overflow: hidden;
}}
.section-header {{
  display: flex; align-items: center; padding: 16px 20px;
  cursor: pointer; user-select: none; gap: 12px;
}}
.section-header:hover {{ background: color-mix(in srgb, var(--accent) 5%, transparent); }}
.section-header .icon {{ font-size: 1.2rem; width: 24px; text-align: center; }}
.section-header .title {{ font-weight: 600; font-size: 1.1rem; flex: 1; }}
.section-header .arrow {{ transition: transform 0.2s; color: var(--text-muted); }}
.section.collapsed .arrow {{ transform: rotate(-90deg); }}
.section.collapsed .section-body {{ display: none; }}
.section-body {{ padding: 0 20px 20px 20px; }}
.chart-container {{ width: 100%; min-height: 400px; }}
.metric-grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px; margin-bottom: 16px;
}}
.metric-card {{
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 16px; text-align: center;
}}
.metric-card .label {{ font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; }}
.metric-card .value {{ font-size: 1.5rem; font-weight: 700; margin-top: 4px; }}
.metric-card .value.pass {{ color: var(--green); }}
.metric-card .value.fail {{ color: var(--red); }}
.metric-card .value.warn {{ color: var(--orange); }}
.status-badge {{
  display: inline-block; padding: 4px 12px; border-radius: 20px;
  font-size: 0.85rem; font-weight: 600;
}}
.status-badge.success {{ background: color-mix(in srgb, var(--green) 20%, transparent); color: var(--green); }}
.status-badge.failure {{ background: color-mix(in srgb, var(--red) 20%, transparent); color: var(--red); }}
.status-badge.warning {{ background: color-mix(in srgb, var(--orange) 20%, transparent); color: var(--orange); }}
table {{
  width: 100%; border-collapse: collapse; margin-top: 8px;
}}
th, td {{
  padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border);
}}
th {{ font-weight: 600; color: var(--text-muted); font-size: 0.85rem; }}
pre {{
  background: var(--bg); padding: 16px; border-radius: 8px;
  overflow-x: auto; font-size: 0.85rem; border: 1px solid var(--border);
}}
@media (max-width: 768px) {{
  body {{ padding: 12px; }}
  .metric-grid {{ grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }}
}}
</style>
</head>
<body>
<button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
<div class="container">
<h1>{title}</h1>
<p class="subtitle">Generated {timestamp}</p>
{sections}
</div>
<script>
function toggleTheme() {{
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  html.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
}}
document.querySelectorAll('.section-header').forEach(header => {{
  header.addEventListener('click', () => {{
    header.parentElement.classList.toggle('collapsed');
  }});
}});
</script>
</body>
</html>"""

_SECTION_TEMPLATE = """<div class="section">
<div class="section-header">
  <span class="icon">{icon}</span>
  <span class="title">{title}</span>
  <span class="arrow">&#9660;</span>
</div>
<div class="section-body">
{body}
</div>
</div>"""


class ReportGenerator:
    def __init__(self, output_dir: Path = Path("reports")) -> None:
        self.output_dir = Path(output_dir)
        self._counter = self._find_next_counter()

    def _find_next_counter(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.output_dir.glob("experiment_*"))
        if not existing:
            return 1
        nums = []
        for d in existing:
            try:
                nums.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return max(nums, default=0) + 1

    def _experiment_dir(self, output_dir: Path | None = None) -> Path:
        base = Path(output_dir) if output_dir is not None else self.output_dir
        exp_dir = base / f"experiment_{self._counter:03d}"
        self._counter += 1
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _has_plotly(self) -> bool:
        try:
            import plotly  # noqa: F401

            return True
        except ImportError:
            return False

    def generate_feasibility_report(self, results: dict, output_dir: Path | None = None) -> Path:
        exp_dir = self._experiment_dir(output_dir)
        sections: list[dict] = []

        overfit = results.get("overfit", {})
        random_label = results.get("random_label", {})
        grad = results.get("gradient_sanity", {})
        act = results.get("activation_stats", {})

        cards = []
        of_success = overfit.get("success", False)
        cards.append(
            f'<div class="metric-card"><div class="label">Overfit Test</div>'
            f'<div class="value {"pass" if of_success else "fail"}">{"PASS" if of_success else "FAIL"}</div></div>'
        )
        of_acc = overfit.get("train_acc", 0)
        cards.append(
            f'<div class="metric-card"><div class="label">Overfit Accuracy</div>'
            f'<div class="value {"pass" if of_acc > 0.9 else "warn"}">{of_acc:.2%}</div></div>'
        )
        rl_mem = random_label.get("can_memorize", False)
        cards.append(
            f'<div class="metric-card"><div class="label">Memorization</div>'
            f'<div class="value {"pass" if rl_mem else "warn"}">{"PASS" if rl_mem else "FAIL"}</div></div>'
        )
        grad_ok = grad.get("gradient_flow_ok", False)
        cards.append(
            f'<div class="metric-card"><div class="label">Gradient Flow</div>'
            f'<div class="value {"pass" if grad_ok else "fail"}">{"OK" if grad_ok else "ISSUE"}</div></div>'
        )
        dead = act.get("dead_neurons_pct", 0)
        cards.append(
            f'<div class="metric-card"><div class="label">Dead Neurons</div>'
            f'<div class="value {"pass" if dead < 10 else "warn"}">{dead:.1f}%</div></div>'
        )

        sections.append(
            {
                "icon": "",
                "title": "Feasibility Summary",
                "body": f'<div class="metric-grid">{"".join(cards)}</div>',
            }
        )

        overfit_body = "<table>"
        for k, v in sorted(overfit.items()):
            if isinstance(v, float):
                overfit_body += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
            else:
                overfit_body += f"<tr><td>{k}</td><td>{v}</td></tr>"
        if overfit_body == "<table>":
            overfit_body += "<tr><td colspan='2'>No data</td></tr>"
        overfit_body += "</table>"

        sections.append(
            {
                "icon": "",
                "title": "Overfit Test Details",
                "body": overfit_body,
            }
        )

        rl_body = "<table>"
        for k, v in sorted(random_label.items()):
            if isinstance(v, float):
                rl_body += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
            else:
                rl_body += f"<tr><td>{k}</td><td>{v}</td></tr>"
        rl_body += "</table>"
        sections.append(
            {
                "icon": "",
                "title": "Random Label Test Details",
                "body": rl_body,
            }
        )

        if grad:
            grad_body = "<table>"
            for k, v in sorted(grad.items()):
                if k == "layer_stats":
                    continue
                if isinstance(v, float):
                    grad_body += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
                else:
                    grad_body += f"<tr><td>{k}</td><td>{v}</td></tr>"
            if "layer_stats" in grad:
                grad_body += (
                    "<tr><td>layer_stats</td><td><pre>" + json.dumps(grad["layer_stats"], indent=2) + "</pre></td></tr>"
                )
            grad_body += "</table>"
            sections.append(
                {
                    "icon": "",
                    "title": "Gradient Sanity Check",
                    "body": grad_body,
                }
            )

        if act:
            act_body = "<table>"
            for k, v in sorted(act.items()):
                if isinstance(v, float):
                    act_body += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
                else:
                    act_body += f"<tr><td>{k}</td><td>{v}</td></tr>"
            act_body += "</table>"
            sections.append(
                {
                    "icon": "",
                    "title": "Activation Statistics",
                    "body": act_body,
                }
            )

        html = self._build_html(sections, "Feasibility Analysis Report")
        out_path = exp_dir / "index.html"
        out_path.write_text(html)
        logger.info("Feasibility report saved to %s", out_path)
        return out_path

    def generate_ablation_report(self, results: dict, output_dir: Path | None = None) -> Path:
        exp_dir = self._experiment_dir(output_dir)
        sections: list[dict] = []

        size_df = results.get("size_ablation")
        class_df = results.get("class_ablation")
        difficulty_df = results.get("difficulty_ablation")
        influence = results.get("influence")

        charts = []

        if size_df is not None:
            try:
                chart = self._plot_ablation_curves(size_df, "fraction", "accuracy")
                if chart:
                    charts.append(chart)
            except Exception as e:
                logger.warning("Failed to generate size ablation chart: %s", e)

        if class_df is not None:
            try:
                chart = self._plot_bar_chart(
                    {
                        "Class": class_df.index.tolist() if hasattr(class_df, "index") else list(range(len(class_df))),
                        "Accuracy": class_df.values.tolist() if hasattr(class_df, "values") else class_df,
                    },
                    "Per-Class Ablation Impact",
                )
                if chart:
                    charts.append(chart)
            except Exception as e:
                logger.warning("Failed to generate class ablation chart: %s", e)

        chart_bodies = ""
        for chart in charts:
            chart_bodies += f'<div class="chart-container" id="chart_{hash(chart) % 100000}"></div>'
            chart_bodies += (
                f"<script>Plotly.newPlot('chart_{hash(chart) % 100000}', {chart}).then(function(){{}});</script>"
            )

        if chart_bodies:
            sections.append(
                {
                    "icon": "",
                    "title": "Ablation Charts",
                    "body": chart_bodies,
                }
            )

        if influence:
            inf_body = "<table><tr><th>Metric</th><th>Value</th></tr>"
            for k, v in sorted(influence.items()):
                if isinstance(v, list):
                    v_repr = f"[{len(v)} samples]"
                elif isinstance(v, float):
                    v_repr = f"{v:.4f}"
                else:
                    v_repr = str(v)
                inf_body += f"<tr><td>{k}</td><td>{v_repr}</td></tr>"
            inf_body += "</table>"
            sections.append(
                {
                    "icon": "",
                    "title": "Influence Scores",
                    "body": inf_body,
                }
            )

        html = self._build_html(sections, "Data Ablation Report")
        out_path = exp_dir / "index.html"
        out_path.write_text(html)
        logger.info("Ablation report saved to %s", out_path)
        return out_path

    def generate_learning_curve_report(
        self, diagnosis: dict, metrics_history: list, output_dir: Path | None = None
    ) -> Path:
        exp_dir = self._experiment_dir(output_dir)
        sections: list[dict] = []

        status = diagnosis.get("status", "unknown")
        status_class = {
            "normal": "success",
            "overfitting": "warning",
            "underfitting": "warning",
            "diverging": "failure",
            "plateau": "warning",
            "high_lr": "failure",
            "low_lr": "warning",
        }.get(status, "warning")

        cards = []
        cards.append(
            f'<div class="metric-card"><div class="label">Status</div>'
            f'<div class="value {status_class}">{status.upper()}</div></div>'
        )
        es = diagnosis.get("early_stop_epoch")
        cards.append(
            f'<div class="metric-card"><div class="label">Early Stop</div>'
            f'<div class="value">{f"Epoch {es}" if es else "N/A"}</div></div>'
        )
        recs = diagnosis.get("recommendations", [])
        cards.append(
            f'<div class="metric-card"><div class="label">Recommendations</div>'
            f'<div class="value {"warn" if recs else "pass"}">{len(recs)}</div></div>'
        )

        sections.append(
            {
                "icon": "",
                "title": "Training Diagnosis",
                "body": f'<div class="metric-grid">{"".join(cards)}</div>',
            }
        )

        if recs:
            rec_body = "<ul style='padding-left:20px;'>"
            for r in recs:
                rec_body += f"<li>{r}</li>"
            rec_body += "</ul>"
            sections.append(
                {
                    "icon": "",
                    "title": "Recommendations",
                    "body": rec_body,
                }
            )

        details = diagnosis.get("details", {})
        if details:
            det_body = "<table>"
            for k, v in sorted(details.items()):
                det_body += f"<tr><td>{k}</td><td>{v}</td></tr>"
            det_body += "</table>"
            sections.append(
                {
                    "icon": "",
                    "title": "Diagnostic Details",
                    "body": det_body,
                }
            )

        if metrics_history:
            try:
                chart = self._plot_metrics(metrics_history, "Learning Curves")
                if chart:
                    chart_id = "learning_curve_chart"
                    chart_body = f'<div class="chart-container" id="{chart_id}"></div>'
                    chart_body += f"<script>Plotly.newPlot('{chart_id}', {chart}).then(function(){{}});</script>"
                    sections.append(
                        {
                            "icon": "",
                            "title": "Learning Curves",
                            "body": chart_body,
                        }
                    )
            except Exception as e:
                logger.warning("Failed to generate learning curve chart: %s", e)

        html = self._build_html(sections, "Learning Curve Analysis Report")
        out_path = exp_dir / "index.html"
        out_path.write_text(html)
        logger.info("Learning curve report saved to %s", out_path)
        return out_path

    def generate_combined_report(self, all_results: dict, output_dir: Path | None = None) -> Path:
        exp_dir = self._experiment_dir(output_dir)
        sections: list[dict] = []

        if "feasibility" in all_results:
            try:
                sub_path = self.generate_feasibility_report(all_results["feasibility"], output_dir=exp_dir)
                sections.append(
                    {
                        "icon": "",
                        "title": "Feasibility Analysis",
                        "body": f'<p>See <a href="{sub_path.name}" target="_blank">full feasibility report</a></p>'
                        f"<pre>{json.dumps(all_results['feasibility'], indent=2, default=str)[:2000]}</pre>",
                    }
                )
            except Exception as e:
                logger.warning("Failed to embed feasibility: %s", e)

        if "ablation" in all_results:
            try:
                sub_path = self.generate_ablation_report(all_results["ablation"], output_dir=exp_dir)
                sections.append(
                    {
                        "icon": "",
                        "title": "Data Ablation",
                        "body": f'<p>See <a href="{sub_path.name}" target="_blank">full ablation report</a></p>',
                    }
                )
            except Exception as e:
                logger.warning("Failed to embed ablation: %s", e)

        if "learning_curve" in all_results:
            diag = all_results["learning_curve"]
            if isinstance(diag, dict):
                try:
                    sub_path = self.generate_learning_curve_report(
                        diag, all_results.get("metrics_history", []), output_dir=exp_dir
                    )
                    sections.append(
                        {
                            "icon": "",
                            "title": "Learning Curve Analysis",
                            "body": f'<p>See <a href="{sub_path.name}" target="_blank">full learning curve report</a></p>',
                        }
                    )
                except Exception as e:
                    logger.warning("Failed to embed learning curves: %s", e)

        if "few_shot" in all_results:
            try:
                fs = all_results["few_shot"]
                fs_body = "<table><tr><th>K-Shots</th><th>Mean Accuracy</th><th>Std</th></tr>"
                for k, v in sorted(fs.items()):
                    if isinstance(v, dict):
                        fs_body += f"<tr><td>{k}</td><td>{v.get('mean', 0):.2%}</td><td>{v.get('std', 0):.2%}</td></tr>"
                fs_body += "</table>"
                sections.append(
                    {
                        "icon": "",
                        "title": "Few-Shot Evaluation",
                        "body": fs_body,
                    }
                )
            except Exception as e:
                logger.warning("Failed to embed few-shot: %s", e)

        if "linear_probe" in all_results:
            try:
                lp = all_results["linear_probe"]
                lp_body = "<table>"
                for k, v in sorted(lp.items()):
                    if isinstance(v, float):
                        lp_body += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
                lp_body += "</table>"
                sections.append(
                    {
                        "icon": "",
                        "title": "Linear Probe Results",
                        "body": lp_body,
                    }
                )
            except Exception as e:
                logger.warning("Failed to embed linear probe: %s", e)

        html = self._build_html(sections, "Combined Analysis Report")
        out_path = exp_dir / "index.html"
        out_path.write_text(html)
        logger.info("Combined report saved to %s", out_path)
        return out_path

    def _build_html(self, sections: list[dict], title: str) -> str:
        if not self._has_plotly():
            return self._build_fallback_html(sections, title)

        section_html = ""
        for sec in sections:
            section_html += _SECTION_TEMPLATE.format(
                icon=sec.get("icon", ""),
                title=sec.get("title", ""),
                body=sec.get("body", ""),
            )

        return _HTML_TEMPLATE.format(
            title=title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sections=section_html,
        )

    def _build_fallback_html(self, sections: list[dict], title: str) -> str:
        body = f"<h1>{title}</h1>"
        body += "<p style='color:orange;'>Plotly is not installed. Install it with: <code>pip install plotly</code></p>"
        body += "<p>Raw data:</p><pre>"
        for sec in sections:
            body += f"\n\n=== {sec.get('title', '')} ===\n{sec.get('body', '')}"
        body += "</pre>"

        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:monospace;max-width:900px;margin:24px auto;padding:0 16px;background:#1e1e2e;color:#cdd6f4;}}pre{{background:#282840;padding:16px;border-radius:8px;overflow-x:auto;}}</style>
</head>
<body>{body}</body>
</html>"""

    def _plot_metrics(self, metrics_history: list, title: str) -> str:
        if not self._has_plotly():
            return json.dumps(metrics_history)

        import plotly.graph_objects as go

        fig = go.Figure()
        keys = set()
        for m in metrics_history:
            keys.update(m.keys())

        train_keys = sorted(
            [k for k in keys if k.startswith("train")], key=lambda x: (0 if "loss" in x.lower() else 1, x)
        )
        val_keys = sorted([k for k in keys if k.startswith("val")], key=lambda x: (0 if "loss" in x.lower() else 1, x))

        epochs = list(range(1, len(metrics_history) + 1))

        colors = ["#89b4fa", "#a6e3a1", "#fab387", "#f38ba8", "#cba6f7", "#94e2d5", "#f9e2af", "#eba0ac"]
        for i, key in enumerate(train_keys + val_keys):
            values = [m.get(key, None) for m in metrics_history]
            values = [v for v in values if v is not None]
            if not values:
                continue
            color = colors[i % len(colors)]
            dash = "dash" if key.startswith("val") else "solid"
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(values) + 1)),
                    y=values,
                    mode="lines",
                    name=key,
                    line=dict(color=color, dash=dash),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Value",
            template="plotly_dark",
            margin=dict(l=40, r=20, t=40, b=40),
        )

        if len(metrics_history) > 1:
            fig.update_xaxes(range=[1, len(metrics_history)])

        return fig.to_json()

    def _plot_ablation_curves(self, df, x_col: str, y_col: str) -> str:
        if not self._has_plotly():
            return json.dumps({"x": list(df[x_col]) if hasattr(df, "columns") else [], "y": []})

        import plotly.graph_objects as go

        x = df[x_col].tolist() if hasattr(df, "columns") else df.get(x_col, [])
        y = df[y_col].tolist() if hasattr(df, "columns") else df.get(y_col, [])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(color="#89b4fa", width=2),
                marker=dict(size=8, color="#89b4fa"),
            )
        )

        fig.update_layout(
            title=f"Ablation: {y_col} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_dark",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig.to_json()

    def _plot_bar_chart(self, data: dict, title: str) -> str:
        if not self._has_plotly():
            return json.dumps(data)

        import plotly.graph_objects as go

        keys = list(data.keys())
        values = list(data.values())
        if len(keys) != 2:
            return ""

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=values[0],
                y=[float(v) for v in values[1]],
                marker_color="#89b4fa",
            )
        )

        fig.update_layout(
            title=title,
            template="plotly_dark",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig.to_json()
