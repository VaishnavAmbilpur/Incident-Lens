# 📂 06 — Datasets

> **Context:** All 4 verified, publicly available post-mortem reports used for indexing and evaluation.  
> **Prerequisite:** [05_directory_structure.md](./05_directory_structure.md)

---

## 6.1 Dataset Overview

IncidentLens indexes **real, publicly available technical disaster reports** — not synthetic data. This ensures the system is tested against the complexity and ambiguity of real production failures.

---

## 6.2 The 4 Post-Mortem Documents

### 📄 1. CrowdStrike Global Outage (July 2024)

| Field | Detail |
|---|---|
| **File** | `data/postmortems/crowdstrike_outage_2024.pdf` |
| **Incident** | Channel File 291 logic error in Falcon sensor content update |
| **Root Cause** | Windows kernel driver `CSAgent.sys` triggered a memory safety failure (bugcheck `0x00000050`) |
| **Key Technical Terms** | `CSAgent.sys`, `Channel File 291`, `WHEA_UNCORRECTABLE_ERROR`, `DRIVER_IRQL_NOT_LESS_OR_EQUAL`, kernel-mode crash |
| **Impact** | ~8.5 million Windows devices worldwide, $5.4B+ estimated losses |
| **Source** | CrowdStrike official Preliminary Post Incident Review (public PDF) |

### 📄 2. Cloudflare Global BGP Route Leak

| Field | Detail |
|---|---|
| **File** | `data/postmortems/cloudflare_bgp_leak.md` |
| **Incident** | BGP route leak causing global traffic misdirection |
| **Root Cause** | Allegheny Technologies (ASN 33154) leaked 20,000+ BGP prefixes through Verizon (ASN 701), who failed to filter |
| **Key Technical Terms** | BGP prefix hijacking, ASN, route leak, RPKI, prefix filtering, AS path |
| **Impact** | Global latency spikes and partial outages for Cloudflare, Amazon, and others |
| **Source** | Cloudflare public blog post |

### 📄 3. AWS US-East-1 Kinesis Outage

| Field | Detail |
|---|---|
| **File** | `data/postmortems/aws_kinesis_outage.pdf` |
| **Incident** | Cascading failure starting from Kinesis Data Streams in us-east-1 |
| **Root Cause** | OS thread limit exhaustion in internal front-end server communications |
| **Key Technical Terms** | Thread starvation, cascading failure, DynamoDB, Cognito, CloudWatch dependency chain |
| **Impact** | Multi-hour outage affecting dozens of downstream AWS services |
| **Source** | AWS post-incident summary (public) |

### 📄 4. GitHub MySQL Replication Split-Brain

| Field | Detail |
|---|---|
| **File** | `data/postmortems/github_mysql_failover.md` |
| **Incident** | Primary MySQL database failover causing data inconsistency |
| **Root Cause** | Network partition triggered automated failover, creating a split-brain condition between old and new primaries |
| **Key Technical Terms** | Split-brain, replication lag, failover timing, network partition, data consistency, Orchestrator |
| **Impact** | GitHub.com degraded service for ~24 hours with data reconciliation required |
| **Source** | GitHub Engineering blog post |

---

## 6.3 Golden Evaluation Dataset

| Field | Detail |
|---|---|
| **File** | `data/eval_golden_dataset.json` |
| **Contents** | 20 curated question-answer pairs with ground-truth citations |
| **Purpose** | Ragas 4-metric evaluation harness (faithfulness, precision, relevance, recall) |
| **Coverage** | At least 5 questions per post-mortem document |
| **Built In** | Phase 4 |

### Example Entry Format
```json
{
  "question": "What Windows kernel driver caused the CrowdStrike global outage?",
  "ground_truth": "CSAgent.sys — a kernel-mode driver in the CrowdStrike Falcon sensor that processed Channel File 291 content updates. A logic error in the content interpreter triggered memory bugcheck 0x00000050.",
  "source_document": "crowdstrike_outage_2024.pdf",
  "source_pages": [3, 7, 12]
}
```

---

## 🔗 Next

→ [07_phase1_foundations.md](./07_phase1_foundations.md) — Phase 1: Environment Setup & Core Foundations.
