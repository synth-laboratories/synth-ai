# Original manual starter: synthetic metric audit

This owned local-only private Git work fulfills the `synthetic-metric-audit` brief. No remote is configured; no real customer data, third-party code, model calls or paid job is used. All original material is offered under CC0-1.0. This is a bounded toy audit, not release-corpus or organic-utility evidence.

The fixed seed is 20260930. Ninety synthetic labels are zero and ten are one. A majority predictor achieves0.90 aggregate accuracy but minority accuracy zero and macro per-class accuracy0.50. A deliberately trivial feature-rule control achieves1.00. This shows how aggregate accuracy hides minority failures; the feature is the label, so no generalization or scientific novelty is claimed.

Run `python3 experiment.py`. Retained inputs, predictions and metrics are checked independently by counting each class and each predictor's correct rows, then repeated from a fresh private checkout. Publication and qualification remain separate owner decisions. No clout award, money, credit or quota reset is promised by preparing or submitting this work.
