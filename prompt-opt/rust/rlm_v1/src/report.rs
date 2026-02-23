use crate::Result;
use crate::models::EvalReport;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Markdown,
    JsonPretty,
}

#[tracing::instrument(skip_all)]
pub fn render_markdown_report(report: &EvalReport) -> String {
    let mut s = String::new();
    s.push_str("# RLM Eval Report\n\n");
    s.push_str(&format!("- report_id: `{}`\n", report.id));
    s.push_str(&format!("- case_id: `{}`\n", report.case.id));
    s.push_str(&format!(
        "- created_at: `{}`\n",
        report.created_at.to_rfc3339()
    ));
    s.push_str(&format!(
        "- total_score: `{:.4}`\n",
        report.outcome.total_score_0_to_1
    ));
    s.push_str(&format!("- passed: `{}`\n", report.outcome.passed));
    s.push_str("\n## Signals\n\n");
    s.push_str("| signal | weight | score |\n");
    s.push_str("|--------|--------|-------|\n");
    for sc in &report.outcome.signal_scores {
        s.push_str(&format!(
            "| {} | {:.4} | {:.4} |\n",
            sc.name, sc.weight.0, sc.score_0_to_1
        ));
    }
    if !report.outcome.reasoning.trim().is_empty() {
        s.push_str("\n## Reasoning\n\n");
        s.push_str(report.outcome.reasoning.trim());
        s.push('\n');
    }
    s
}

#[tracing::instrument(skip_all)]
pub fn render_report(report: &EvalReport, format: ReportFormat) -> Result<String> {
    match format {
        ReportFormat::Markdown => Ok(render_markdown_report(report)),
        ReportFormat::JsonPretty => Ok(serde_json::to_string_pretty(report)?),
    }
}
