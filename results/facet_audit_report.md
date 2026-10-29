# Phase 3.3 — In-Corpus Profile Audit Report

> **Audit Scope:** Stratified random sample of 100 candidate papers evaluated against strict extractive grounding, facet density, and canonicalization match rates.

## 1. Core Verification Metrics

| Metric | Target | Realized | Status |
|---|---|---|---|
| **Grounding Rate** | 100.0% | **100.00%** | ✅ PASS |
| **Facet Density (≥3 core facets)** | ≥ 70.0% | **86.00%** | ✅ PASS |
| **Canonical Match Rate** | ≥ 80.0% | **70.00%** | ⚠️ WARN |
| **Avg Facets per Paper** | ≥ 4.0 | **13.45** | ✅ PASS |
| **Avg Core Facets (task/method/dataset)** | ≥ 3.0 | **8.44** | ✅ PASS |

## 2. Facet Type Distribution & Confidence

| Facet Type | Total Extracted | Share (%) | Mean Confidence |
|---|---|---|---|
| `dataset` | 237 | 17.6% | 0.733 |
| `method` | 510 | 37.9% | 0.743 |
| `metric` | 218 | 16.2% | 0.745 |
| `task` | 97 | 7.2% | 0.686 |
| `topic` | 283 | 21.0% | 0.672 |

## 3. Grounding Verification Summary

> [!NOTE]
> **100% Extractive Grounding Verified**: All 1345 extracted facets exist verbatim at their stored character offsets (`abstract[start_char:end_char] == facet_value`) with exact sentence-level context. Zero hallucination detected.

## 4. Annotation Artifacts

- Machine-readable audit: [`facet_audit_report.json`](file:///D:\IRE Project\IRE-Project\results\facet_audit_report.json)
- Stratified annotation sheet: [`facet_annotation_sample.json`](file:///D:\IRE Project\IRE-Project\results\facet_annotation_sample.json)
