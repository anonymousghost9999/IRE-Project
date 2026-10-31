# Phase 3.3 — In-Corpus Profile Audit Report

> **Audit Scope:** Stratified random sample of 100 candidate papers evaluated against strict extractive grounding, facet density, and canonicalization match rates.

## 1. Core Verification Metrics

| Metric | Target | Realized | Status |
|---|---|---|---|
| **Grounding Rate** | 100.0% | **100.00%** | ✅ PASS |
| **Facet Density (≥3 core facets)** | ≥ 70.0% | **64.00%** | ⚠️ WARN |
| **Canonical Match Rate** | ≥ 80.0% | **78.00%** | ⚠️ WARN |
| **Avg Facets per Paper** | ≥ 4.0 | **9.68** | ✅ PASS |
| **Avg Core Facets (task/method/dataset)** | ≥ 3.0 | **4.84** | ✅ PASS |

## 2. Facet Type Distribution & Confidence

| Facet Type | Total Extracted | Share (%) | Mean Confidence |
|---|---|---|---|
| `dataset` | 104 | 10.7% | 0.689 |
| `method` | 334 | 34.5% | 0.768 |
| `metric` | 205 | 21.2% | 0.740 |
| `task` | 46 | 4.8% | 0.684 |
| `topic` | 279 | 28.8% | 0.683 |

## 3. Grounding Verification Summary

> [!NOTE]
> **100% Extractive Grounding Verified**: All 968 extracted facets exist verbatim at their stored character offsets (`abstract[start_char:end_char] == facet_value`) with exact sentence-level context. Zero hallucination detected.

## 4. Annotation Artifacts

- Machine-readable audit: [`facet_audit_report.json`](facet_audit_report.json)
- Stratified annotation sheet: [`facet_annotation_sample.json`](facet_annotation_sample.json)
