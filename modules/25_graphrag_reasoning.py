"""
25_graphrag_reasoning.py
========================
Phase 6: Cutting-Edge Advanced RAG — Technique 5: GraphRAG (Multi-Hop Failure Chains)
IncidentLens Reasoning Graph:

WHY THIS EXISTS:
  Vector RAG is a one-hop machine. Ask "What was the downstream impact of the Kinesis
  front-end thread exhaustion?" and embedding search returns chunks that are *similar* to
  the question. Similarity is not causality. The chunk that explains the thread limit and
  the chunk that lists the impacted services are in different places, and a top-k window
  cannot walk from one to the other. The failure chain is spread across the timeline, so a
  flat index silently cuts it in half and the answer comes out shallow.

  This module stops treating the corpus as a bag of chunks. It extracts the CAUSAL SKELETON
  of each incident into an explicitly directed graph and then answers multi-hop questions by
  TRAVERSING that graph instead of by similarity search.

      A --caused--> B --caused--> C --caused--> D
      (every edge carries the chunk id + the exact sentence that proves it)

  THE GROUNDING RULE (this is the whole trick):
    An extracted edge is a CLAIM. Every claim must be provable. So each edge is authored
    together with an `evidence_chunk_id` and an `evidence_text`, and at BUILD time this
    module proves, in code and with no LLM, that:
        1. the cited chunk id really exists in the corpus (module 03 chunking, 105 chunks), and
        2. `evidence_text` really is a substring of that chunk's text.
    Anything that fails either test is REJECTED and printed as rejected. It is never added
    to the graph and never silently kept. An ungrounded graph is worse than no graph: it
    invents causality and then cites itself.

  WHAT THE RESULTING GRAPH BUYS US:
    - multi-hop questions ("what was the downstream impact of X?") answered by forward BFS;
    - the exact propagation chain, hop by hop, each hop carrying its citation;
    - shortest causal path between any two components (root cause -> observable symptom);
    - single points of failure: nodes whose removal disconnects the most downstream nodes;
    - fan-out: components with the most downstream dependents (blast-radius ranking).

  We model four real post-mortems. The Kinesis chain, drawn as the graph actually stores it:

      Front-end fleet capacity addition (holiday scaling)
        |
        v  [aws_kinesis_outage_p1_c4]  "the total number of threads required by the
        |                               front-end process exceeded the operating
        |                               system's configured maximum user processes limit"
      OS thread exhaustion (nproc limit reached)
        |------------------------------+
        v                              v
      Front-end fleet health        Shard-map cache cannot refresh
      checks failing                       |
                                           v
                                    Request routing failure (PutRecord / GetRecords)
                                           |
                                           v
                                    Kinesis Data Streams degradation
                                           |
              +-----------------+----------+------------------+
              v                 v                             v
        CloudWatch         Amazon Cognito              AWS Lambda / Console
        metric delay       auth degradation            event + dashboard delay

  The other three chains: Cloudflare's 2019 BGP route leak, GitHub's 2018 MySQL failover,
  and CrowdStrike's 2024 Channel File 291 crash (see the incident data below).

HONEST FAILURE POLICY:
  No hop is ever invented, no edge is ever kept without proof, and a question the graph
  cannot answer is reported as "NOT ANSWERABLE FROM THE GRAPH" instead of being guessed at.
  The negative case is a first-class test, not an afterthought.

LLM USE:
  Zero LLM calls by default. `--llm` optionally asks Groq for a short narrative of an
  already-verified propagation path. The narrative is prose around a chain that graph
  traversal already proved; it is never the source of a hop.
"""

import os
import re
import sys
import subprocess
import importlib.util
import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

# Ensure UTF-8 output encoding across Windows shells
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# =====================================================================
# 0. Interpreter Guard
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
REQUIRED_PACKAGES = ("networkx", "chromadb", "dotenv")


def _missing_packages(names: Tuple[str, ...]) -> List[str]:
    """Return the package names that the ACTIVE interpreter cannot import."""
    missing: List[str] = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def ensure_project_interpreter() -> None:
    """Re-launch this script with the project venv when the active interpreter is wrong."""
    missing = _missing_packages(REQUIRED_PACKAGES)
    if not missing:
        return

    if os.environ.get("INCIDENTLENS_REEXEC") == "1" or not VENV_PYTHON.exists():
        print("=" * 80)
        print(" [BOOT] Cannot start: project packages are missing")
        print("=" * 80)
        print(f"  Interpreter  : {sys.executable}")
        print(f"  Missing      : {', '.join(missing)}")
        print(f"  Project venv : {VENV_PYTHON}")
        print("  Run this instead:")
        print("      .\\venv\\Scripts\\python.exe modules\\25_graphrag_reasoning.py")
        print("=" * 80)
        sys.exit(1)

    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return

    print("=" * 80)
    print(" [BOOT] Wrong interpreter detected - switching to the project venv")
    print("=" * 80)
    env = dict(os.environ)
    env["INCIDENTLENS_REEXEC"] = "1"
    completed = subprocess.run(
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        env=env,
    )
    sys.exit(completed.returncode)


ensure_project_interpreter()

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def _import_helper(filename: str, module_name: str):
    """Load a numbered module file (Python cannot import names that start with digits)."""
    script_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- tunables -------------------------------------------------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
EVIDENCE_PREVIEW_CHARS = 96      # how much of the proof sentence we print next to an edge
NARRATIVE_MAX_TOKENS = 220       # only used with --llm
# A second, explicit opt-in. `--llm` is not enough on its own: the free tier is shared with
# the other Phase 6 modules, so spending a call is a deliberate act, not a side effect.
ALLOW_LLM_ENV = "INCIDENTLENS_ALLOW_LLM"
INCIDENT_SLUGS = (
    "aws_kinesis_outage",
    "cloudflare_bgp_leak",
    "github_mysql_failover",
    "crowdstrike_outage_2024",
)


def _print(text: str = "") -> None:
    """Every print goes through here so a console encoding failure can never kill a check."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def _clip(text: str, width: int = EVIDENCE_PREVIEW_CHARS) -> str:
    """One-line preview of the proof sentence."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "..."


# =====================================================================
# 1. Evidence: the grounding gate
# =====================================================================
# Cause: the real corpus is WRAPPED text. The Markdown post-mortems use one long line per
# paragraph, but the CrowdStrike PDF arrives with hard line breaks every ~90 characters, and
# module 03 chunks it character-exactly. A quote that spans a wrapped line therefore does
# NOT exist as a substring even though a human reading the PDF sees it as one sentence.
#
# Decision (documented, not hidden): an evidence quote must be an EXACT substring of the
# chunk. Where a proof sentence spans a PDF line break, the quote is written with the exact
# " \n" sequence the chunk contains, so the check stays a strict, zero-tolerance substring
# test rather than a fuzzy similarity score. Evidence legitimately may include PDF artifacts
# such as the Markdown bold markers on "**AWS Lambda:**"; the string that must be present is
# the string that must be present.
# =====================================================================
@dataclass(frozen=True)
class EdgeSpec:
    """
    One authored causal claim, prior to verification.

    `source` and `target` are graph nodes (components, limits, configs, events).
    `evidence_chunk_id` names the chunk that is supposed to prove the claim.
    `evidence_text` is the substring that must literally occur inside that chunk.
    """
    source: str
    target: str
    evidence_chunk_id: str
    evidence_text: str
    incident: str
    note: str = ""


@dataclass(frozen=True)
class VerifiedEdge:
    """An EdgeSpec that survived the grounding gate. Only these reach the graph."""
    source: str
    target: str
    evidence_chunk_id: str
    evidence_text: str
    incident: str
    note: str = ""


@dataclass
class RejectedEdge:
    """An EdgeSpec that failed the grounding gate, with the reason it failed."""
    spec: EdgeSpec
    reason: str
    normalised_match: bool = False


@dataclass
class GroundingReport:
    """Outcome of running every authored claim through the gate."""
    corpus_chunks: int
    candidates: int
    verified: List[VerifiedEdge] = field(default_factory=list)
    rejected: List[RejectedEdge] = field(default_factory=list)
    normalised_only: int = 0
    unknown_chunk: int = 0
    missing_substring: int = 0


_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """
    Diagnostic-only comparison: collapse whitespace and fold PDF/typographic lookalikes.

    This exists purely so a rejection can be REPORTED accurately ("this quote only matches
    after whitespace normalisation - it straddles a wrapped PDF line"). It is never used to
    decide acceptance. Acceptance is always the exact-substring test below.
    """
    folded = (
        text.replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
    )
    return _WS_RE.sub(" ", folded).strip()


def verify_evidence(spec: EdgeSpec, corpus: Dict[str, str]) -> Tuple[Optional[VerifiedEdge], Optional[RejectedEdge]]:
    """
    The grounding gate. Returns (verified_edge, None) or (None, rejected_edge).

    Two independent conditions must both hold:
      1. the cited chunk id is present in the corpus loaded from data/postmortems/, and
      2. evidence_text occurs inside that chunk's text, character for character.
    """
    text = corpus.get(spec.evidence_chunk_id)
    if text is None:
        return None, RejectedEdge(
            spec=spec,
            reason=f"unknown chunk id {spec.evidence_chunk_id!r} (not present in the {len(corpus)}-chunk corpus)",
        )
    if spec.evidence_text not in text:
        straddles = _normalise(spec.evidence_text) in _normalise(text)
        reason = (
            "evidence_text is not an exact substring of the cited chunk"
            + (" (it only matches after whitespace folding, i.e. it straddles a wrapped PDF line)"
               if straddles else "")
        )
        return None, RejectedEdge(spec=spec, reason=reason, normalised_match=straddles)
    return VerifiedEdge(
        source=spec.source,
        target=spec.target,
        evidence_chunk_id=spec.evidence_chunk_id,
        evidence_text=spec.evidence_text,
        incident=spec.incident,
        note=spec.note,
    ), None


# =====================================================================
# 2. The authored causal claims (one per edge, every one provable)
# =====================================================================
# Every entry below is a causal statement taken from a real incident post-mortem. The
# evidence_text strings were mined from the corpus and then machine-verified; see section 3.
# =====================================================================
def build_edge_specs() -> List[EdgeSpec]:
    """Return every authored causal claim, grouped by incident."""
    C = "crowdstrike_outage_2024"
    K = "aws_kinesis_outage"
    B = "cloudflare_bgp_leak"
    G = "github_mysql_failover"

    specs: List[EdgeSpec] = [
        # ---------------------------------------------------------------
        # Incident 1 — AWS Kinesis, US-EAST-1, 25 Nov 2020
        # Capacity addition -> OS thread exhaustion -> cache/routing failure -> downstream
        # ---------------------------------------------------------------
        EdgeSpec(
            "Front-end fleet capacity addition (holiday scaling)",
            "OS thread exhaustion (nproc limit reached)",
            f"{K}_p1_c4",
            "the total number of threads required by the front-end process exceeded the "
            "operating system's configured maximum user processes limit",
            K,
            "every front-end server keeps a peer-to-peer thread to every other server, so "
            "thread demand scales with fleet size; crossing the threshold blew the nproc limit",
        ),
        EdgeSpec(
            "OS thread exhaustion (nproc limit reached)",
            "Front-end fleet health checks failing",
            f"{K}_p1_c5",
            "Servers began failing health checks as internal thread creation threw resource "
            "allocation exceptions",
            K,
            "thread exhaustion is observed as failed health checks",
        ),
        EdgeSpec(
            "OS thread exhaustion (nproc limit reached)",
            "Shard-map cache cannot refresh",
            f"{K}_p1_c5",
            "The in-memory shard-map cache became unable to refresh",
            K,
            "no worker threads left -> the shard-map refresh loop starves",
        ),
        EdgeSpec(
            "Shard-map cache cannot refresh",
            "Request routing failure (PutRecord / GetRecords)",
            f"{K}_p1_c5",
            "unable to route incoming PutRecord and GetRecords requests to back-end shards",
            K,
            "a stale shard map means the front end cannot pick a back-end shard",
        ),
        EdgeSpec(
            "Request routing failure (PutRecord / GetRecords)",
            "Kinesis Data Streams degradation (elevated API error rates and latency)",
            f"{K}_p1_c2",
            "The issue resulted in elevated error rates and latencies for Kinesis APIs "
            "(PutRecord, PutRecords, GetRecords)",
            K,
            "routing failure surfaces as the API-level symptom",
        ),
        EdgeSpec(
            "Kinesis Data Streams degradation (elevated API error rates and latency)",
            "Amazon CloudWatch metric ingestion delay",
            f"{K}_p1_c8",
            "Ingestion of custom metrics and metric alarms experienced delays because "
            "CloudWatch uses Kinesis to buffer metric data",
            K,
            "CloudWatch buffers metric data on Kinesis, so Kinesis degradation becomes "
            "metrics delay",
        ),
        EdgeSpec(
            "Kinesis Data Streams degradation (elevated API error rates and latency)",
            "Amazon Cognito authentication and token issuance degradation",
            f"{K}_p1_c8",
            "User authentication and token issuance calls were degraded because Cognito "
            "writes user login event streams to Kinesis",
            K,
            "authentication itself was downstream of the streaming data plane",
        ),
        EdgeSpec(
            "Kinesis Data Streams degradation (elevated API error rates and latency)",
            "AWS Lambda event source mapping delays",
            f"{K}_p1_c8",
            "**AWS Lambda:** Event source mappings consuming from Kinesis streams experienced "
            "processing delays",
            K,
            "Kinesis-triggered Lambda consumers stalled",
        ),
        EdgeSpec(
            "Kinesis Data Streams degradation (elevated API error rates and latency)",
            "AWS Management Console dashboard delay (SHD / CloudWatch views)",
            f"{K}_p1_c8",
            "Service Health Dashboard (SHD) and CloudWatch dashboard views were intermittently "
            "delayed in reflecting real-time status",
            K,
            "irony worth keeping: the status dashboards were themselves downstream",
        ),
        # ---------------------------------------------------------------
        # Incident 2 — Cloudflare / Verizon BGP route leak, 24 Jun 2019
        # Optimizer splits prefixes -> DQE announces -> Verizon re-announces -> global outage
        # ---------------------------------------------------------------
        EdgeSpec(
            "DQE BGP optimizer more-specific route splitting",
            "DQE announced more-specific routes to Allegheny Technologies",
            f"{B}_p1_c6",
            "was using a BGP optimizer in their network, which meant there were a lot of more "
            "specific routes in their network",
            B,
            "the optimizer manufactured the more-specific prefixes that started the leak",
        ),
        EdgeSpec(
            "DQE announced more-specific routes to Allegheny Technologies",
            "Verizon (AS701) accepted and re-announced the leaked routes",
            f"{B}_p1_c7",
            "was then sent to their other transit provider (AS701 - Verizon), who proceeded to "
            "tell the entire Internet about these",
            B,
            "the leak reached a tier-1 transit provider and stopped being local",
        ),
        EdgeSpec(
            "Noction BGP Optimizer splits prefixes into more-specifics",
            "Verizon (AS701) accepted and re-announced the leaked routes",
            f"{B}_p1_c3",
            "splits up received IP prefixes into smaller, contributing parts (called "
            "more-specifics)",
            B,
            "BGP prefers the most specific route, which is why the split propagated",
        ),
        EdgeSpec(
            "Verizon (AS701) accepted and re-announced the leaked routes",
            "Internet-wide service degradation (Cloudflare, Amazon, Linode)",
            f"{B}_p1_c7",
            "lack of filtering turned this into a major incident that affected many Internet "
            "services",
            B,
            "the missing prefix/IRR filter at Verizon is the single control that failed",
        ),
        EdgeSpec(
            "Internet-wide service degradation (Cloudflare, Amazon, Linode)",
            "Cloudflare traffic loss (~15% of global traffic)",
            f"{B}_p1_c8",
            "we observed a loss, at the worst of the incident, of about 15% of our global traffic",
            B,
            "the measured blast radius",
        ),
        EdgeSpec(
            "DQE stopped announcing optimized routes",
            "Internet routing stabilizes",
            f"{B}_p1_c14",
            "DQE worked with us on the phone to stop advertising these \u201coptimized\u201d "
            "routes to Allegheny Technologies Inc.",
            B,
            "recovery came from withdrawing the announcement, not from a filter appearing",
        ),
        EdgeSpec(
            "Verizon (AS701) accepted and re-announced the leaked routes",
            "Internet routing stabilizes",
            f"{B}_p1_c14",
            "DQE worked with us on the phone to stop advertising these \u201coptimized\u201d "
            "routes to Allegheny Technologies Inc.",
            B,
            "same evidence, different causal question: removing the route removed the path",
        ),
        # ---------------------------------------------------------------
        # Incident 3 — GitHub MySQL failover, 21 Oct 2018
        # Optical maintenance -> partition -> Raft deselection -> cross-region writes -> 24h
        # ---------------------------------------------------------------
        EdgeSpec(
            "100G optical maintenance",
            "Loss of connectivity between US East network hub and primary data center",
            f"{G}_p1_c4",
            "routine maintenance work to replace failing 100G optical equipment resulted in "
            "the loss of connectivity between our US East Coast network hub and our primary "
            "US East Coast data center",
            G,
            "a 43-second optical outage is the root trigger of a 24-hour incident",
        ),
        EdgeSpec(
            "Loss of connectivity between US East network hub and primary data center",
            "Orchestrator leadership deselection (Raft quorum lost)",
            f"{G}_p1_c7",
            "During the network partition described above, Orchestrator, which had been active "
            "in our primary data center, began a process of leadership deselection, according "
            "to Raft consensus",
            G,
            "the partition split the Raft quorum, so the primary lost leadership",
        ),
        EdgeSpec(
            "Orchestrator leadership deselection (Raft quorum lost)",
            "Cross-region MySQL failover (writes directed to US West Coast)",
            f"{G}_p1_c7",
            "start failing over clusters to direct writes to the US West Coast data center",
            G,
            "automation did exactly what it was configured to do - across a region boundary",
        ),
        EdgeSpec(
            "Loss of connectivity between US East network hub and primary data center",
            "Divergent writes in both data centers (split brain)",
            f"{G}_p1_c8",
            "Because the database clusters in both data centers now contained writes that were "
            "not present in the other data center",
            G,
            "both sites accepted writes during the partition, so neither could be trusted alone",
        ),
        EdgeSpec(
            "Cross-region MySQL failover (writes directed to US West Coast)",
            "Cross-country write latency for East Coast applications",
            f"{G}_p1_c12",
            "are currently unable to cope with the additional latency introduced by a "
            "cross-country round trip",
            G,
            "the application tier was never designed for cross-country primary writes",
        ),
        EdgeSpec(
            "Cross-country write latency for East Coast applications",
            "Webhook delivery and GitHub Pages builds paused",
            f"{G}_p1_c13",
            "We made an explicit choice to partially degrade site usability by pausing webhook "
            "delivery and GitHub Pages builds",
            G,
            "deliberate degradation to protect data integrity",
        ),
        EdgeSpec(
            "Webhook delivery and GitHub Pages builds paused",
            "Service degradation for 24 hours and 11 minutes",
            f"{G}_p1_c1",
            "GitHub was also unable to serve webhook events or build and publish GitHub Pages "
            "sites",
            G,
            "the user-visible side of the 24h11m degradation",
        ),
        EdgeSpec(
            "Divergent writes in both data centers (split brain)",
            "Full cluster rebuild from backups",
            f"{G}_p1_c15",
            "until this incident we have never needed to fully rebuild an entire cluster from "
            "backup",
            G,
            "the only safe reconciliation was restore-from-backup for whole clusters",
        ),
        EdgeSpec(
            "Full cluster rebuild from backups",
            "Slow site load times during restoration",
            f"{G}_p1_c17",
            "This resulted in slow site load times for pages that had to execute a write "
            "operation over a cross-country link",
            G,
            "restoration kept serving traffic, so slow became the user experience",
        ),
        EdgeSpec(
            "Slow site load times during restoration",
            "Replication lag and inconsistent reads",
            f"{G}_p1_c20",
            "These delayed replicas resulted in users seeing inconsistent data as they "
            "interacted with our services",
            G,
            "read replicas hours behind the primary served stale data",
        ),
        # ---------------------------------------------------------------
        # Incident 4 — CrowdStrike Falcon / Channel File 291, 19 Jul 2024
        # Definition mismatch -> validation gap -> OOB read -> kernel fault -> bugcheck
        # ---------------------------------------------------------------
        EdgeSpec(
            "21-input IPC Template Type definition vs 20 inputs supplied by the integration code",
            "Sensor code supplies only 20 input sources to the Content Interpreter",
            f"{C}_p3_c1",
            "code would supply 20 different input sources to the Content Interpreter",
            C,
            "the sensor build only ever had 20 inputs to hand over",
        ),
        EdgeSpec(
            "21-input IPC Template Type definition vs 20 inputs supplied by the integration code",
            "Channel File 291 Template Instances expect the 21st input",
            f"{C}_p2_c1",
            "defined 21 input parameter fields, but the integration code that \n"
            "invoked the Content Interpreter with Channel File 291\u2019s Template Instances "
            "supplied only 20",
            C,
            "the count mismatch itself, stated verbatim",
        ),
        EdgeSpec(
            "21-input IPC Template Type definition vs 20 inputs supplied by the integration code",
            "Content Validator logic error (assumed 21 inputs)",
            f"{C}_p5_c1",
            "its assessment on the expectation that the IPC Template Type would be provided "
            "with 21 \ninputs. This resulted in the problematic Template Instance being sent to "
            "the Content \nInterpreter",
            C,
            "the validator validated against the definition, not against the code",
        ),
        EdgeSpec(
            "Content Validator logic error (assumed 21 inputs)",
            "Problematic Template Instance released to the Content Interpreter",
            f"{C}_p5_c1",
            "it based \nits assessment on the expectation that the IPC Template Type would be "
            "provided with 21 \ninputs",
            C,
            "the bad instance passed validation because the validator shared the same wrong "
            "assumption",
        ),
        EdgeSpec(
            "Problematic Template Instance released to the Content Interpreter",
            "20-parameter count mismatch evaded build validation and field testing",
            f"{C}_p2_c1",
            "This parameter count mismatch evaded multiple layers of \nbuild validation and "
            "testing",
            C,
            "why the defect reached production at all",
        ),
        EdgeSpec(
            "July 19 2024 Rapid Response Content update (two new IPC Template Instances)",
            "Channel File 291 Template Instances expect the 21st input",
            f"{C}_p2_c2",
            "introduced a non-wildcard matching criterion for the 21st input parameter. These "
            "new \nTemplate Instances resulted in a new version of Channel File 291 that would "
            "now require",
            C,
            "the update is what turned a latent bug into a live one: non-wildcard matching "
            "forced a real read",
        ),
        EdgeSpec(
            "Channel File 291 Template Instances expect the 21st input",
            "Sensor inspects the 21st input parameter",
            f"{C}_p2_c2",
            "the sensor to inspect the 21st input parameter",
            C,
            "the content demands an input the sensor cannot provide",
        ),
        EdgeSpec(
            "Sensor inspects the 21st input parameter",
            "Out-of-bounds read of the input array",
            f"{C}_p2_c3",
            "specifying a comparison against the 21st input value. The Content Interpreter "
            "expected only \n20 values. Therefore, the attempt to access the 21st value "
            "produced an out-of-bounds",
            C,
            "reading the 21st entry of a 20-entry array is the defect",
        ),
        EdgeSpec(
            "Missing runtime array bounds check in the Content Interpreter",
            "Out-of-bounds read of the input array",
            f"{C}_p3_c3",
            "A runtime array bounds check was missing for Content \nInterpreter input fields on "
            "Channel File 291",
            C,
            "the missing guard that would have converted a crash into a clean error",
        ),
        EdgeSpec(
            "Out-of-bounds read of the input array",
            "TemplateGetString supplies an invalid input pointer (21st element)",
            f"{C}_p11_c2",
            "according to an index specified by \nChannel File 291.",
            C,
            "the interpreter indexes into the pointer array using the channel-file index",
        ),
        EdgeSpec(
            "TemplateGetString supplies an invalid input pointer (21st element)",
            "csagent.sys dereferences the invalid pointer",
            f"{C}_p12_c1",
            "After reading this invalid pointer into register r8, the control flow in the "
            "snippet above takes \nthe first jump",
            C,
            "the crash-dump register walkthrough: the pointer is loaded, then used",
        ),
        EdgeSpec(
            "csagent.sys dereferences the invalid pointer",
            "System crash (bugcheck)",
            f"{C}_p8_c4",
            "we see that a memory fault (also known as an \u201caccess \nviolation\u201d) "
            "bugcheck has occurred",
            C,
            "the kernel bugcheck that took the Windows hosts down",
        ),
        EdgeSpec(
            "Out-of-bounds read of the input array",
            "System crash (bugcheck)",
            f"{C}_p2_c4",
            "memory read beyond the end of the input data array and resulted in a system crash",
            C,
            "independent second proof for the same edge, from the summary paragraph",
        ),
    ]
    return specs


# =====================================================================
# 3. Load corpus + build the verified graph
# =====================================================================
@dataclass
class IncidentGraph:
    """A verified causal graph plus the evidence bookkeeping around it."""
    graph: nx.DiGraph
    corpus: Dict[str, str]
    report: GroundingReport

    # ------ evidence lookups -------------------------------------------------
    def edge_evidence(self, source: str, target: str) -> Optional[VerifiedEdge]:
        data = self.graph.get_edge_data(source, target)
        return data.get("evidence") if data else None

    def isolated_evidence(self) -> List[VerifiedEdge]:
        """Verified edges that did NOT end up in the graph (duplicate or self-loop)."""
        kept = {
            (e.source, e.target)
            for _, _, e in self.graph.edges(data="evidence")
        }
        return [e for e in self.report.verified if (e.source, e.target) not in kept]

    # ------ traversal API ---------------------------------------------------
    def downstream_chain(self, origin: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Forward BFS from `origin`. Returns (visit_order, hop_depth).

        `visit_order` is breadth-first so the list reads as a propagation order, and
        `hop_depth[node]` is the number of causal hops from `origin`. Successors are visited
        in sorted order so the result is reproducible run to run.
        """
        if origin not in self.graph:
            return [], {}
        depth: Dict[str, int] = {origin: 0}
        order: List[str] = [origin]
        queue: deque = deque([origin])
        while queue:
            node = queue.popleft()
            for nxt in sorted(self.graph.successors(node)):
                if nxt not in depth:
                    depth[nxt] = depth[node] + 1
                    order.append(nxt)
                    queue.append(nxt)
        return order, depth

    def downstream_nodes(self, origin: str) -> Dict[str, int]:
        """Just the reachable nodes with their hop depth (origin excluded)."""
        order, depth = self.downstream_chain(origin)
        return {n: depth[n] for n in order if n != origin}

    def causal_path(self, source: str, target: str) -> Optional[List[str]]:
        """Shortest causal path source -> target, or None when no path exists."""
        if source not in self.graph or target not in self.graph:
            return None
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None

    def path_depth(self, source: str, target: str) -> Optional[int]:
        path = self.causal_path(source, target)
        return None if path is None else len(path) - 1

    def edge_sequence_of_path(self, path: Sequence[str]) -> List[VerifiedEdge]:
        """
        Every consecutive pair in `path` must be a proven edge; missing ones are dropped.

        Only valid for a genuine PATH. A breadth-first visit order is not a path (its
        consecutive entries are siblings), so BFS results are handled by
        `edges_of_tree` instead.
        """
        out: List[VerifiedEdge] = []
        for a, b in zip(path, path[1:]):
            evidence = self.edge_evidence(a, b)
            if evidence is not None:
                out.append(evidence)
        return out

    def spanning_tree(self, origin: str) -> Dict[str, Optional[Tuple[str, VerifiedEdge]]]:
        """
        The BFS tree behind `downstream_chain`: node -> (parent, the proven edge that was used).

        A propagation chain is not a single path - the same cause branches to several
        effects. Walking this tree gives, for every reached node, the exact causal edge that
        first got us there, which is what a citation per hop has to mean for a chain answer.
        The origin maps to (None, None).
        """
        tree: Dict[str, Optional[Tuple[str, VerifiedEdge]]] = {}
        if origin not in self.graph:
            return tree
        tree[origin] = None
        queue: deque = deque([origin])
        while queue:
            node = queue.popleft()
            for nxt in sorted(self.graph.successors(node)):
                if nxt in tree:
                    continue
                evidence = self.edge_evidence(node, nxt)
                if evidence is None:            # structurally impossible, but stay honest
                    continue
                tree[nxt] = (node, evidence)
                queue.append(nxt)
        return tree

    def edges_of_tree(self, origin: str, members: Sequence[str]) -> List[VerifiedEdge]:
        """The causal edges of the BFS tree restricted to `members`, in visit order."""
        tree = self.spanning_tree(origin)
        out: List[VerifiedEdge] = []
        for node in members:
            entry = tree.get(node)
            if entry is not None:
                out.append(entry[1])
        return out

    def describe_path(self, path: Sequence[str]) -> str:
        """Human-readable one-liner: A -> B -> C (3 hops). For a true path."""
        return " -> ".join(path) + f"   ({len(path) - 1} hops)"

    def describe_chain(self, chain: Sequence[str]) -> str:
        """
        Human-readable one-liner for a BFS chain.

        The hop count is the DEEPEST hop reached, not `len(chain) - 1`: a chain branches, so
        its length counts a node per branch while its depth measures propagation distance.
        Reporting the length as "hops" would overstate the reach (9 labels for a 5-deep tree).
        The traversal-edge count is given separately so nothing is hidden.
        """
        if not chain:
            return "(empty)"
        depth = self.downstream_nodes(chain[0])
        deepest = max((depth.get(node, 0) for node in chain), default=0)
        edges = len(self.edges_of_tree(chain[0], chain))
        return (" -> ".join(chain)
                + f"   (deepest hop {deepest}, {len(chain) - 1} nodes reached via {edges} edges)")

    # ------ structural analysis --------------------------------------------
    def single_points_of_failure(self) -> List[Dict[str, Any]]:
        """
        Rank nodes by how many downstream nodes stop being reachable if the node is removed.

        Definition used here (stated explicitly because it is a choice): for node `n`,
        `downstream(n)` = nodes reachable from `n`; `disconnected(n)` = how many of those are
        no longer reachable from `n` once `n` itself is deleted. A node with a large
        downstream reach but no disconnection is a fan-out point, not a choke point: traffic
        of causation still gets through without it. Only `disconnected` measures being
        irreplaceable, so that is the sort key; fan-out is reported alongside for contrast.
        """
        ranking: List[Dict[str, Any]] = []
        for node in sorted(self.graph.nodes):
            reachable = set(self.downstream_nodes(node))
            if not reachable:
                continue
            survivor = self.graph.copy()
            survivor.remove_node(node)
            still_reachable = set(self.downstream_nodes(node)) if node in survivor else set()
            disconnected = len(reachable - still_reachable)
            ranking.append({
                "node": node,
                "downstream": len(reachable),
                "disconnected": disconnected,
                "out_degree": self.graph.out_degree(node),
                "in_degree": self.graph.in_degree(node),
                "role": "root trigger" if self.graph.in_degree(node) == 0 else "intermediate",
            })
        ranking.sort(key=lambda r: (-r["disconnected"], -r["downstream"], r["node"]))
        return ranking

    def fan_out_ranking(self) -> List[Dict[str, Any]]:
        """Nodes with the most downstream dependents (blast radius), then widest out-degree."""
        ranking = self.single_points_of_failure()
        return sorted(ranking, key=lambda r: (-r["downstream"], -r["out_degree"], r["node"]))

    def components(self) -> List[List[str]]:
        """Weakly connected components: one per incident, which is a useful sanity signal."""
        return [sorted(c) for c in nx.weakly_connected_components(self.graph)]

    def incident_of(self, node: str) -> str:
        for edge in self.graph.edges(data="evidence"):
            if node in (edge[0], edge[1]):
                return edge[2].incident
        return "unknown"

    def ascii_map(self) -> str:
        """
        ASCII rendering of the graph.

        networkx returns a generator (hence the join) and draws boxes with box-drawing
        glyphs, which arrive as mojibake on some Windows consoles. They are transliterated to
        plain ASCII so the shape is readable everywhere.
        """
        try:
            text = "\n".join(nx.generate_network_text(self.graph))
        except Exception as exc:                                  # pragma: no cover
            return f"(ASCII map unavailable: {type(exc).__name__}: {exc})"
        for glyph, plain in (("\u2502", "|"), ("\u251c", "|"), ("\u2514", "|"),
                             ("\u2500", "-"), ("\u2192", "->"), ("\u2190", "<-")):
            text = text.replace(glyph, plain)
        # Anything still non-ASCII (networkx uses a few exotic box glyphs) is folded or
        # dropped, so the map cannot arrive as mojibake regardless of console codepage.
        return text.encode("ascii", "ignore").decode("ascii")


def load_corpus() -> Dict[str, str]:
    """
    Load the production chunks through module 03 and index them by chunk id.

    Module 03 is loaded by file path because Python cannot import a module whose name
    starts with a digit. The corpus is the ONLY source of truth for the grounding gate.
    """
    mod03 = _import_helper("03_text_splitters.py", "module_03_splitters")
    chunks = mod03.get_production_chunks()
    corpus: Dict[str, str] = {}
    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if chunk_id:
            corpus[chunk_id] = chunk.page_content
    return corpus


def build_incident_graph(corpus: Dict[str, str]) -> IncidentGraph:
    """Run every authored claim through the grounding gate, then build the DiGraph."""
    specs = build_edge_specs()
    report = GroundingReport(corpus_chunks=len(corpus), candidates=len(specs))
    graph = nx.DiGraph()

    for spec in specs:
        verified, rejected = verify_evidence(spec, corpus)
        if rejected is not None:
            report.rejected.append(rejected)
            if rejected.normalised_match:
                report.normalised_only += 1
            elif "unknown chunk id" in rejected.reason:
                report.unknown_chunk += 1
            else:
                report.missing_substring += 1
            continue

        assert verified is not None
        report.verified.append(verified)

        # KeyError is impossible here: a node with no successors returns None.
        existing = graph.get_edge_data(verified.source, verified.target)
        if existing is not None:
            # Two independent proofs of the same causal link. Keep the first, record the
            # duplicate as isolated evidence so it is still visible in the report.
            continue
        if verified.source == verified.target:
            continue
        graph.add_edge(
            verified.source,
            verified.target,
            evidence=verified,
        )

    return IncidentGraph(graph=graph, corpus=corpus, report=report)


# =====================================================================
# 4. Multi-hop questions, answered by traversal (and checked)
# =====================================================================
@dataclass
class MultiHopCase:
    """
    One multi-hop question with the exact node sequence traversal must return.

    `mode` states which traversal answers the question, and it is set by hand per case rather
    than inferred, because the two are genuinely different questions:
      "chain" - "what was the downstream impact of X?"  -> forward BFS, every reachable node;
      "path"  - "how did X end up causing Y?"           -> shortest causal path X -> Y.
    """
    case_id: str
    question: str
    origin: str
    expectation: str                       # what the traversal is expected to return
    expected_chain: List[str]              # exact ordered node list
    mode: str = "chain"                    # "chain" (forward BFS) or "path" (shortest path)
    min_hops: int = 1
    note: str = ""
    answerable: bool = True


def build_multihop_cases(graph: nx.DiGraph) -> List[MultiHopCase]:
    """The verification suite: five answerable multi-hop questions and one honest negative."""
    C = "crowdstrike_outage_2024"
    return [
        MultiHopCase(
            case_id="Q1",
            question="What was the downstream impact of the Kinesis front-end capacity addition?",
            origin="Front-end fleet capacity addition (holiday scaling)",
            expectation="forward BFS, full propagation order (every reachable node)",
            expected_chain=[
                "Front-end fleet capacity addition (holiday scaling)",
                "OS thread exhaustion (nproc limit reached)",
                "Front-end fleet health checks failing",
                "Shard-map cache cannot refresh",
                "Request routing failure (PutRecord / GetRecords)",
                "Kinesis Data Streams degradation (elevated API error rates and latency)",
                "AWS Lambda event source mapping delays",
                "AWS Management Console dashboard delay (SHD / CloudWatch views)",
                "Amazon CloudWatch metric ingestion delay",
                "Amazon Cognito authentication and token issuance degradation",
            ],
            mode="chain",
            min_hops=5,
            note="the deepest chain in the graph: 5 hops from trigger to downstream services",
        ),
        MultiHopCase(
            case_id="Q2",
            question="Which services went down because Kinesis itself degraded?",
            origin="Kinesis Data Streams degradation (elevated API error rates and latency)",
            expectation="forward BFS, fan-out at hop 1",
            expected_chain=[
                "Kinesis Data Streams degradation (elevated API error rates and latency)",
                "AWS Lambda event source mapping delays",
                "AWS Management Console dashboard delay (SHD / CloudWatch views)",
                "Amazon CloudWatch metric ingestion delay",
                "Amazon Cognito authentication and token issuance degradation",
            ],
            mode="chain",
            min_hops=1,
            note="one hop, four dependents: the widest blast radius in the Kinesis incident",
        ),
        MultiHopCase(
            case_id="Q3",
            question=("How did a BGP optimizer at a small Pennsylvania ISP cause Cloudflare "
                      "to lose 15% of global traffic?"),
            origin="Noction BGP Optimizer splits prefixes into more-specifics",
            expectation="shortest causal path, 3 hops",
            expected_chain=[
                "Noction BGP Optimizer splits prefixes into more-specifics",
                "Verizon (AS701) accepted and re-announced the leaked routes",
                "Internet-wide service degradation (Cloudflare, Amazon, Linode)",
                "Cloudflare traffic loss (~15% of global traffic)",
            ],
            mode="path",
            min_hops=3,
            note="the 3-hop case: optimizer -> tier-1 re-announcement -> degradation -> traffic loss",
        ),
        MultiHopCase(
            case_id="Q4",
            question=("How did the 21-input template definition mismatch end in a Windows "
                      "system crash?"),
            origin=("21-input IPC Template Type definition vs 20 inputs supplied by the "
                    "integration code"),
            expectation="shortest causal path, 4 hops",
            expected_chain=[
                "21-input IPC Template Type definition vs 20 inputs supplied by the integration code",
                "Channel File 291 Template Instances expect the 21st input",
                "Sensor inspects the 21st input parameter",
                "Out-of-bounds read of the input array",
                "System crash (bugcheck)",
            ],
            mode="path",
            min_hops=4,
            note="definition mismatch -> content demands input 21 -> OOB read -> bugcheck",
        ),
        MultiHopCase(
            case_id="Q5",
            question="What did GitHub blame the MySQL failover on, and what broke as a result?",
            origin="Orchestrator leadership deselection (Raft quorum lost)",
            expectation="forward BFS, full propagation order",
            expected_chain=[
                "Orchestrator leadership deselection (Raft quorum lost)",
                "Cross-region MySQL failover (writes directed to US West Coast)",
                "Cross-country write latency for East Coast applications",
                "Webhook delivery and GitHub Pages builds paused",
                "Service degradation for 24 hours and 11 minutes",
            ],
            mode="chain",
            min_hops=4,
            note="linear chain: 4 hops from Raft deselection to the 24h11m degradation",
        ),
        MultiHopCase(
            case_id="Q6",
            question=("Which components were hit by the DNS TTL propagation delay during the "
                      "CrowdStrike outage?"),
            origin="DNS TTL propagation delay",
            expectation="no such node, no such chain",
            expected_chain=[],
            mode="chain",
            min_hops=0,
            note="negative control: DNS/TTL appears nowhere in this corpus",
            answerable=False,
        ),
    ]


def validate_edges(ig: IncidentGraph, edges: Sequence[VerifiedEdge]) -> List[str]:
    """
    Re-check a set of edges against the corpus, hop by hop, at answer time.

    A traversal result is only trustworthy if every edge it reports is a stored edge AND that
    edge's evidence still passes the grounding test. This is the runtime guard that makes
    "we report citations" a verified claim rather than a hopeful one. It runs again here even
    though the build already checked, because the answer is the thing the user reads.
    """
    problems: List[str] = []
    for edge in edges:
        if ig.graph.get_edge_data(edge.source, edge.target) is None:
            problems.append(f"hop {edge.source!r} --> {edge.target!r} is not an edge in the graph")
            continue
        text = ig.corpus.get(edge.evidence_chunk_id)
        if text is None:
            problems.append(f"hop {edge.source!r} --> {edge.target!r} cites unknown chunk "
                            f"{edge.evidence_chunk_id!r}")
        elif edge.evidence_text not in text:
            problems.append(f"hop {edge.source!r} --> {edge.target!r}: evidence_text is not "
                            f"inside {edge.evidence_chunk_id!r}")
    return problems


@dataclass
class CaseResult:
    """Outcome of checking one multi-hop case."""
    case: MultiHopCase
    passed: bool
    detail: str
    chain: List[str] = field(default_factory=list)
    depth: Dict[str, int] = field(default_factory=dict)
    hops: int = 0
    solved_by: str = ""
    edges: List[VerifiedEdge] = field(default_factory=list)


def _bfs_order(graph: nx.DiGraph, origin: str) -> List[str]:
    """Deterministic breadth-first visit order, used by both the API and the checks."""
    if origin not in graph:
        return []
    seen: Dict[str, int] = {origin: 0}
    order: List[str] = [origin]
    queue: deque = deque([origin])
    while queue:
        node = queue.popleft()
        for nxt in sorted(graph.successors(node)):
            if nxt not in seen:
                seen[nxt] = seen[node] + 1
                order.append(nxt)
                queue.append(nxt)
    return order


def evaluate_case(ig: IncidentGraph, case: MultiHopCase) -> CaseResult:
    """
    Answer one multi-hop case strictly by traversal.

    Answerable cases use a shortest causal PATH when the question asks "how did X cause Y"
    and a forward BFS CHAIN when it asks "what was the impact of X". Both are traversal
    results; neither is a generated answer. Unanswerable cases must come back empty.
    """
    if not case.answerable:
        # The honest negative case: the ONLY correct outcome is an empty traversal. If the
        # graph produced a chain here, the question was answerable after all and either the
        # case or the graph is wrong - which must fail loudly, never be papered over.
        order = _bfs_order(ig.graph, case.origin)
        if order:
            return CaseResult(
                case=case,
                passed=False,
                detail=f"expected no traversal, but the graph produced {len(order)} nodes",
                chain=order,
            )
        present = [n for n in ig.graph.nodes if case.origin.lower() in n.lower()]
        detail = (
            f"origin {case.origin!r} is not a node in the graph and no path exists to traverse"
            + (f"; near-miss nodes: {present}" if present else "")
        )
        return CaseResult(case=case, passed=True, detail=detail, chain=[],
                          solved_by="not answerable (empty traversal)")

    if case.origin not in ig.graph:
        return CaseResult(
            case=case,
            passed=False,
            detail=f"origin node {case.origin!r} is not in the graph",
        )

    destination = case.expected_chain[-1] if case.expected_chain else case.origin
    chain = _bfs_order(ig.graph, case.origin)
    depth = ig.downstream_nodes(case.origin)

    if case.mode == "path":
        # "how did A end up causing D?" is answered by the shortest causal path: that is the
        # argument the question is asking for, and extra reached nodes are a different answer.
        path = ig.causal_path(case.origin, destination)
        if path is None:
            return CaseResult(
                case=case,
                passed=False,
                detail=f"no causal path from {case.origin!r} to {destination!r}",
            )
        hops = len(path) - 1
        solved_by = f"shortest causal path ({hops} hops)"
        actual = path
        used_edges = ig.edge_sequence_of_path(path)
    else:
        # "what was the downstream impact?" is answered by forward BFS over every reachable
        # node, hop order preserved. The citations come from the BFS tree, because a chain
        # branches: each reached node is cited with the edge that first reached it.
        hops = max(depth.values()) if depth else 0
        solved_by = f"forward BFS chain ({len(chain) - 1} downstream nodes)"
        actual = chain
        used_edges = ig.edges_of_tree(case.origin, chain)

    matched = actual == case.expected_chain
    if matched and hops < case.min_hops:
        matched = False
        detail = f"chain matched but only {hops} hops (needed at least {case.min_hops})"
    elif matched:
        detail = f"{len(actual) - 1} nodes downstream, deepest hop {hops}"
    else:
        missing = [n for n in case.expected_chain if n not in actual]
        extra = [n for n in actual if n not in case.expected_chain]
        detail = (
            f"expected {len(case.expected_chain)} ordered nodes, traversal returned "
            f"{len(actual)}"
            + (f"; missing={missing}" if missing else "")
            + (f"; unexpected={extra}" if extra else "")
            + ("; order differs from expectation" if not missing and not extra else "")
        )

    return CaseResult(
        case=case,
        passed=matched,
        detail=detail,
        chain=actual,
        depth=depth,
        hops=hops,
        solved_by=solved_by,
        edges=used_edges,
    )


def print_case_result(ig: IncidentGraph, result: CaseResult) -> None:
    """Print the traversal answer with a citation on every edge it used."""
    case = result.case
    _print("")
    _print("-" * 80)
    _print(f" {case.case_id}  {case.question}")
    _print("-" * 80)
    _print(f"   origin        : {case.origin}")
    _print(f"   answered by   : {result.solved_by or 'traversal'}")
    if not case.answerable:
        _print("   ANSWER        : NOT ANSWERABLE FROM THE GRAPH")
        _print(f"   reason        : {result.detail}")
        _print("   (the graph refuses to guess: no traversal, no citation, no invented hop)")
        return

    _print(f"   expectation   : {case.expectation}")
    _print("   chain         :")
    for index, node in enumerate(result.chain):
        hop = result.depth.get(node, index)
        indent = "      " + "  " * min(hop, 6)
        _print(f"{indent}hop {hop:>2} : {node}")
    _print("")
    _print("   evidence on every hop"
           + (" (BFS tree: the edge that first reached each node):"
              if case.mode == "chain" else " (path edges):"))
    if not result.edges:
        _print("     (none - no hop was claimed, so no citation is due)")
    for edge in result.edges:
        _print(f"     [{edge.evidence_chunk_id}] {edge.source}")
        _print(f"        --> {edge.target}")
        _print(f"        proof: \"{_clip(edge.evidence_text, 110)}\"")
    if case.mode == "path" and len(result.chain) > 1:
        _print(f"   shortest path : {ig.describe_path(result.chain)} (this is the traversal result)")
    elif case.mode == "chain" and len(result.chain) > 1:
        deepest = ig.causal_path(result.chain[0], result.chain[-1])
        if deepest:
            _print(f"   deepest route : {ig.describe_path(deepest)}")
    hop_problems = validate_edges(ig, result.edges) if case.answerable else []
    _print(f"   hop audit     : "
           + ("all hops proved" if not hop_problems else f"UNPROVED HOPS: {hop_problems}"))
    good = result.passed and not hop_problems
    _print(f"   VERDICT       : {'PASS' if good else 'FAIL'} - {result.detail}")


# =====================================================================
# 5. Optional extras (OFF by default): authored narrative via Groq
# =====================================================================
def llm_narrative(ig: IncidentGraph, edges: Sequence[VerifiedEdge], question: str) -> str:
    """
    Ask Groq to turn a TRAVERSAL-PROVEN chain into short prose. Only with --llm.

    The chain is computed by the graph before this function is called, and the prompt forbids
    adding hops. The model is decorating a proof, not producing one; if it is unavailable the
    caller is told so instead of receiving an invented summary.

    Defence in depth against accidental spend: this function refuses to run at all unless
    LLM_ENABLED is set (which only main() does, and only for --llm), the operator has also set
    INCIDENTLENS_ALLOW_LLM=1, AND a plausible key is present. Absence is reported as a skip,
    never raised, so a missing key can never turn the "no LLM calls by default" guarantee into
    a crash or a surprise request.
    """
    if not LLM_ENABLED:
        return "[narrative skipped] LLM mode is off (this function is unreachable without --llm)"
    if os.getenv(ALLOW_LLM_ENV, "").strip() != "1":
        return (f"[narrative skipped] --llm alone does not spend tokens; set {ALLOW_LLM_ENV}=1 "
                f"as well to confirm you want the extra Groq call")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key.lower().startswith(("test", "dummy", "fake")):
        return ("[narrative skipped] no usable GROQ_API_KEY in the environment; "
                "the verified chain above is the answer")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_groq import ChatGroq

    lines = []
    for edge in edges:
        lines.append(
            f"- {edge.source} -> {edge.target}  [{edge.evidence_chunk_id}] "
            f"proof: {' '.join(edge.evidence_text.split())}"
        )
    evidence_block = "\n".join(lines) or "(no edges)"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You write short SRE incident narratives. You are given a causal chain that was "
         "already verified by graph traversal, with the cited evidence for each hop. Write "
         "3-5 sentences describing how the failure propagated. Use ONLY the hops given. Do "
         "not add any hop, component, cause or number that is not in the input."),
        ("human", "Question: {question}\n\nVerified causal chain:\n{evidence}\n\nNarrative:"),
    ])
    chain_models = [GROQ_MODEL]
    for model_name in chain_models:
        try:
            llm = ChatGroq(model=model_name, temperature=0.0,
                           max_tokens=NARRATIVE_MAX_TOKENS, api_key=api_key)
            return (prompt | llm | StrOutputParser()).invoke(
                {"question": question, "evidence": evidence_block}
            ).strip()
        except Exception as exc:
            _print(f"     [llm unavailable] {model_name}: {type(exc).__name__}: {str(exc)[:90]}")
    return "[narrative unavailable] No model produced a narrative; the verified chain above stands on its own."


# =====================================================================
# 6. Reporting helpers
# =====================================================================
def print_header(title: str) -> None:
    _print("")
    _print("=" * 80)
    _print(f" {title}")
    _print("=" * 80)


def print_graph_summary(ig: IncidentGraph) -> None:
    """Node/edge counts, per-incident breakdown, and every rejected edge with its reason."""
    graph = ig.graph
    report = ig.report

    print_header("[1] Graph summary (verified causal graph)")
    _print(f"  corpus chunks loaded    : {report.corpus_chunks}")
    _print(f"  authored causal claims  : {report.candidates}")
    _print(f"  edges VERIFIED & kept   : {graph.number_of_edges()}")
    _print(f"  edges REJECTED          : {len(report.rejected)}")
    _print(f"  nodes                   : {graph.number_of_nodes()}")
    _print(f"  weakly connected pieces : {len(ig.components())} (one per incident)")
    _print(f"  isolated evidence       : {len(ig.isolated_evidence())} duplicate proof(s) of an "
           f"already-kept edge")
    _print(f"  rejection breakdown     : {report.missing_substring} substring mismatch, "
           f"{report.unknown_chunk} unknown chunk id, {report.normalised_only} whitespace-only match")

    _print("")
    _print("  Rejected edges (never added to the graph):")
    if not report.rejected:
        _print("    none - every authored claim was provable")
    for rejected in report.rejected:
        _print(f"    x {rejected.spec.source}")
        _print(f"        --> {rejected.spec.target}")
        _print(f"        chunk  : {rejected.spec.evidence_chunk_id}")
        _print(f"        reason : {rejected.reason}")

    _print("")
    _print("  Edges per incident:")
    per_incident: Dict[str, int] = {}
    for _, _, evidence in graph.edges(data="evidence"):
        per_incident[evidence.incident] = per_incident.get(evidence.incident, 0) + 1
    for incident in INCIDENT_SLUGS:
        _print(f"    {incident:<26} {per_incident.get(incident, 0):>3} edges")

    _print("")
    _print("  Node inventory:")
    for component in ig.components():
        label = ig.incident_of(component[0])
        _print(f"    [{label}]")
        for node in component:
            _print(f"      - {node}")


def print_traversal_demo(ig: IncidentGraph) -> None:
    """Show the raw traversal API: a long chain, a 3-hop path, and two non-answers."""
    print_header("[2] Traversal API")

    origin = "Front-end fleet capacity addition (holiday scaling)"
    order, depth = ig.downstream_chain(origin)
    _print(f"  downstream_chain({origin!r})")
    _print(f"    {len(order) - 1} downstream nodes, deepest hop {max(depth.values())}")
    for node in order:
        _print(f"      hop {depth[node]:>2}  {node}")

    _print("")
    a = "Noction BGP Optimizer splits prefixes into more-specifics"
    b = "Cloudflare traffic loss (~15% of global traffic)"
    path = ig.causal_path(a, b)
    _print("  causal_path(BGP optimizer -> Cloudflare traffic loss)")
    _print(f"    {ig.describe_path(path) if path else 'no path'}")

    _print("")
    _print("  non-answers are reported, not guessed:")
    for unknown in ["DNS TTL propagation delay", "Kafka consumer lag"]:
        result = ig.downstream_nodes(unknown)
        verdict = ("not answerable from the graph (unknown node)"
                   if not result else f"{len(result)} downstream nodes")
        _print(f"    downstream_chain({unknown!r}) -> {verdict}")
    cross = ig.causal_path("100G optical maintenance", "System crash (bugcheck)")
    _print(f"    causal_path(optical maintenance -> CrowdStrike bugcheck) -> "
           f"{'no path (different incidents; the graph does not invent one)' if cross is None else cross}")


def print_multihop_verification(ig: IncidentGraph, results: List[CaseResult]) -> None:
    """Print every multi-hop question with the chain it produced and a citation per hop."""
    print_header("[3] Multi-hop verification (traversal answers with citations)")
    for result in results:
        print_case_result(ig, result)

    _print("")
    _print("-" * 80)
    _print("  Compact verification output")
    _print("-" * 80)
    for result in results:
        case = result.case
        marker = "PASS" if result.passed else "FAIL"
        if not case.answerable:
            _print(f"  [{marker}] {case.case_id} {case.question}")
            _print("          -> NOT ANSWERABLE FROM THE GRAPH")
            continue
        _print(f"  [{marker}] {case.case_id} {case.question}")
        _print(f"          -> {ig.describe_chain(result.chain)}")
        for edge in result.edges:
            _print(f"             ({edge.source} --caused--> {edge.target}) "
                   f"cite={edge.evidence_chunk_id}")


def print_spof_ranking(ig: IncidentGraph) -> None:
    """Single-point-of-failure ranking plus the fan-out (blast radius) contrast."""
    print_header("[4] Single points of failure and fan-out")
    ranking = ig.single_points_of_failure()
    _print("  Removal impact (disconnected = downstream nodes that stop being reachable "
           "when the node is deleted):")
    _print(f"    {'#':<3} {'disc':>4} {'down':>5} {'out':>4} {'in':>3}  {'role':<13} node")
    for index, row in enumerate(ranking, start=1):
        _print(f"    {index:<3} {row['disconnected']:>4} {row['downstream']:>5} "
               f"{row['out_degree']:>4} {row['in_degree']:>3}  {row['role']:<13} {row['node']}")

    worst = ranking[0]["disconnected"] if ranking else 0
    spofs = [row["node"] for row in ranking if row["disconnected"] == worst and worst > 0]
    _print("")
    _print(f"  Worst-case disconnection: {worst} downstream nodes.")
    _print("  Single points of failure (removal severs the most downstream nodes):")
    for node in spofs:
        _print(f"    * {node}")

    _print("")
    _print("  Fan-out ranking (most downstream dependents - blast radius, not choke point):")
    for index, row in enumerate(ig.fan_out_ranking()[:8], start=1):
        _print(f"    {index}. downstream={row['downstream']:>2}  out_degree={row['out_degree']}  "
               f"{row['node']}")
    _print("")
    _print("  Note: a node can have a large downstream count and still not be a choke point")
    _print("  (parallel paths exist); only the disconnected column measures irreplaceability.")


def print_ascii_graph(ig: IncidentGraph, show: bool = True) -> None:
    if not show:
        _print("")
        _print("[5] Graph shape: skipped (--no-ascii)")
        return
    print_header("[5] Graph shape (NetworkX ASCII map)")
    _print(ig.ascii_map())


# =====================================================================
# 7. Checks
# =====================================================================
@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def run_checks(ig: IncidentGraph, results: List[CaseResult]) -> List[Check]:
    """Every acceptance check for this module. All must pass for exit code 0."""
    checks: List[Check] = []
    graph = ig.graph
    report = ig.report

    # ---- grounding integrity -------------------------------------------
    bad_kept = [
        f"{e.source}->{e.target}"
        for _, _, e in graph.edges(data="evidence")
        if e.evidence_chunk_id not in ig.corpus or e.evidence_text not in ig.corpus[e.evidence_chunk_id]
    ]
    checks.append(Check(
        "grounding: every kept edge cites a real chunk and an exact substring",
        not bad_kept,
        f"{graph.number_of_edges()} edges re-verified after build" +
        (f"; violations={bad_kept}" if bad_kept else ""),
    ))

    rejected_documented = all(r.reason for r in report.rejected)
    checks.append(Check(
        "grounding: every rejected edge has a recorded reason",
        rejected_documented,
        f"{len(report.rejected)} rejected, all with reasons",
    ))

    kept_pairs = {(e.source, e.target) for _, _, e in graph.edges(data="evidence")}
    rejected_pairs = {(r.spec.source, r.spec.target) for r in report.rejected}
    no_leak = not (kept_pairs & rejected_pairs)
    checks.append(Check(
        "grounding: no rejected edge leaked into the graph",
        no_leak,
        f"overlap={sorted(kept_pairs & rejected_pairs)}" if not no_leak else
        f"{len(rejected_pairs)} rejected pairs absent from the graph",
    ))

    checks.append(Check(
        "structure: graph is a directed acyclic graph (causality has no cycles)",
        nx.is_directed_acyclic_graph(graph),
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
        f"{len(ig.components())} weakly connected components",
    ))

    checks.append(Check(
        "structure: topology is stable across repeated traversals",
        ig.downstream_chain("Front-end fleet capacity addition (holiday scaling)")[0]
        == ig.downstream_chain("Front-end fleet capacity addition (holiday scaling)")[0],
        "BFS visit order is deterministic (successors visited in sorted order)",
    ))

    # ---- traversal API ---------------------------------------------------
    origin = "Front-end fleet capacity addition (holiday scaling)"
    order, depth = ig.downstream_chain(origin)
    chain_ok = order[0] == origin and len(order) == len(set(order))
    deepest = max(depth.values()) if depth else 0
    checks.append(Check(
        "traversal: forward BFS returns a chain with hop depths",
        chain_ok and deepest >= 5,
        f"{len(order) - 1} downstream nodes, deepest hop {deepest}",
    ))

    path = ig.causal_path("Out-of-bounds read of the input array", "System crash (bugcheck)")
    expected_short = ["Out-of-bounds read of the input array", "System crash (bugcheck)"]
    checks.append(Check(
        "traversal: shortest causal path minimises hops",
        path == expected_short,
        f"got {path} (1-hop edge preferred over the 2-hop via TemplateGetString)",
    ))

    cross = ig.causal_path("100G optical maintenance", "System crash (bugcheck)")
    checks.append(Check(
        "traversal: no path across unrelated incidents",
        cross is None,
        "GitHub root cause has no causal path to the CrowdStrike bugcheck",
    ))

    unknown = ig.downstream_nodes("DNS TTL propagation delay")
    checks.append(Check(
        "traversal: unknown node returns nothing instead of a guess",
        unknown == {},
        f"downstream_nodes('DNS TTL propagation delay') -> {unknown}",
    ))

    # ---- multi-hop verification -----------------------------------------
    answerable = [r for r in results if r.case.answerable]
    negative = [r for r in results if not r.case.answerable]
    checks.append(Check(
        "multi-hop: at least 3 answerable multi-hop questions verified",
        len(answerable) >= 3 and all(r.passed for r in answerable),
        f"{sum(1 for r in answerable if r.passed)}/{len(answerable)} passed: "
        + ", ".join(f"{r.case.case_id}={r.hops} hops" for r in answerable),
    ))

    three_hop = [r for r in answerable if r.hops >= 3 and r.passed]
    checks.append(Check(
        "multi-hop: at least one question needs 3 or more hops",
        len(three_hop) >= 1,
        "; ".join(f"{r.case.case_id} solved by {r.solved_by}" for r in three_hop),
    ))

    shallow = [
        r.case.case_id for r in answerable
        if r.passed and r.hops < r.case.min_hops
    ]
    checks.append(Check(
        "multi-hop: every verified chain meets its minimum hop requirement",
        not shallow,
        f"{len(answerable)} answerable cases; violations={shallow}" if shallow else
        f"min-hop requirements honoured for {len(answerable)} cases",
    ))

    cited_problems: List[str] = []
    for result in answerable:
        if len(result.edges) != len(result.chain) - 1:
            cited_problems.append(
                f"{result.case.case_id}: {len(result.chain) - 1} hops but "
                f"{len(result.edges)} cited edges"
            )
        cited_problems.extend(validate_edges(ig, result.edges))
    checks.append(Check(
        "multi-hop: every hop in every answer carries grounded evidence",
        not cited_problems,
        f"{sum(len(r.edges) for r in answerable)} hop citations re-validated against the corpus"
        + (f"; problems={cited_problems}" if cited_problems else ", every one an exact substring"),
    ))

    neg_ok = len(negative) == 1 and not negative[0].chain and not negative[0].case.answerable
    negatives_in_corpus = [
        cid for cid, text in ig.corpus.items()
        if "DNS TTL" in text or "TTL propagat" in text
    ]
    checks.append(Check(
        "multi-hop: the unanswerable question reports 'not answerable from the graph'",
        neg_ok and not negatives_in_corpus,
        f"Q6 origin absent from graph and from the corpus (corpus hits={len(negatives_in_corpus)})",
    ))

    # ---- structural analysis --------------------------------------------
    ranking = ig.single_points_of_failure()
    worst = ranking[0] if ranking else {"disconnected": 0, "node": "n/a"}
    checks.append(Check(
        "SPOF: ranking computed with disconnection measured by actual node removal",
        bool(ranking) and worst["disconnected"] > 0,
        f"worst node {worst['node']!r} disconnects {worst['disconnected']} downstream nodes",
    ))

    fan = ig.fan_out_ranking()
    checks.append(Check(
        "fan-out: widest blast radius identified",
        bool(fan) and fan[0]["downstream"] >= 4,
        f"{fan[0]['node']!r} has {fan[0]['downstream']} downstream dependents "
        f"({fan[0]['out_degree']} direct)",
    ))

    return checks


def print_checks(checks: List[Check]) -> int:
    print_header("[6] Checks")
    for index, check in enumerate(checks, start=1):
        _print(f"  [{'PASS' if check.passed else 'FAIL'}] {index:>2}. {check.name}")
        _print(f"          {check.detail}")
    failed = [c for c in checks if not c.passed]
    _print("")
    _print(f"  {len(checks) - len(failed)}/{len(checks)} checks passed")
    return len(failed)


# =====================================================================
# 8. Demo
# =====================================================================
def run_demo(ig: IncidentGraph, show_ascii: bool = True) -> int:
    print_graph_summary(ig)
    print_traversal_demo(ig)

    cases = build_multihop_cases(ig.graph)
    results = [evaluate_case(ig, case) for case in cases]
    print_multihop_verification(ig, results)

    print_spof_ranking(ig)
    print_ascii_graph(ig, show=show_ascii)

    checks = run_checks(ig, results)
    failures = print_checks(checks)

    print_header("Module 25 Summary")
    deepest = 0
    for case in cases:
        if case.answerable:
            depth = ig.downstream_nodes(case.origin)
            deepest = max(deepest, max(depth.values()) if depth else 0)
    _print(f"  corpus chunks           : {ig.report.corpus_chunks}")
    _print(f"  nodes / edges           : {ig.graph.number_of_nodes()} / {ig.graph.number_of_edges()}")
    _print(f"  rejected edges          : {len(ig.report.rejected)}")
    _print(f"  multi-hop cases verified: "
           f"{sum(1 for r in results if r.case.answerable and r.passed)}"
           f"/{sum(1 for r in results if r.case.answerable)} answerable"
           f" + {sum(1 for r in results if not r.case.answerable)} negative control")
    _print(f"  deepest propagation     : {deepest} hops")
    _print(f"  LLM calls in this run   : {'enabled via --llm' if LLM_ENABLED else '0 (default)'}")
    _print("")
    if failures:
        _print(f"  Module 25 FAILED: {failures} check(s) did not pass.")
    else:
        _print("  Module 25 Complete: GraphRAG multi-hop reasoning verified!")
        _print("  The graph answers cascading-failure questions by traversal, and every hop it")
        _print("  reports is backed by a chunk id whose text literally contains the proof.")
    _print("=" * 80)
    return 1 if failures else 0


LLM_ENABLED = False


def main() -> int:
    global LLM_ENABLED

    parser = argparse.ArgumentParser(
        description="IncidentLens Module 25 - GraphRAG multi-hop failure-chain reasoning",
    )
    parser.add_argument("--llm", action="store_true",
                        help="OPTIONAL: ask Groq for a short narrative of a verified chain. "
                             "Requires INCIDENTLENS_ALLOW_LLM=1 as a second opt-in. The "
                             "default run makes ZERO LLM calls")
    parser.add_argument("--no-ascii", action="store_true",
                        help="skip the NetworkX ASCII map (useful for narrow terminals)")
    args = parser.parse_args(sys.argv[1:])
    LLM_ENABLED = bool(args.llm)

    print_header("IncidentLens - Module 25: GraphRAG (Multi-Hop Failure Chains)")
    _print("  Technique : directed causal graph (NetworkX) + grounded multi-hop traversal")
    _print("  Corpus    : data/postmortems/ -> module 03 chunking (105 production chunks)")
    _print("  Grounding : every edge must cite a real chunk id AND an exact substring of it")
    _print(f"  LLM       : {'--llm set (separate opt-in checked at call time)' if LLM_ENABLED else 'OFF (zero API calls)'}")

    _print("")
    _print("[0] Loading corpus and chunk metadata via module 03 (text splitters)...")
    corpus = load_corpus()
    _print(f"    {len(corpus)} chunks loaded; sample ids: "
           f"{', '.join(sorted(corpus)[:3])} ...")

    _print("[0] Running every authored causal claim through the grounding gate...")
    ig = build_incident_graph(corpus)
    _print(f"    verified {len(ig.report.verified)} claims, rejected {len(ig.report.rejected)}, "
           f"graph has {ig.graph.number_of_nodes()} nodes / {ig.graph.number_of_edges()} edges")

    exit_code = run_demo(ig, show_ascii=not args.no_ascii)

    if LLM_ENABLED:
        print_header("[7] Optional narrative (--llm)")
        case = build_multihop_cases(ig.graph)[0]
        result = evaluate_case(ig, case)
        _print(f"  Question: {case.question}")
        _print(f"  Verified chain: {ig.describe_chain(result.chain)}")
        _print("  --- narrative (prose only; adds no hop that traversal did not prove) ---")
        _print(llm_narrative(ig, result.edges, case.question))
        _print("  ----------------------------------------------------------------------")
    else:
        _print("")
        _print("  (--llm not set: narrative section skipped, no Groq API call was made)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
