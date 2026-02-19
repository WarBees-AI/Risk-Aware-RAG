# RAI-RAG Architecture

RAI-RAG (Risk-Aware Introspective Retrieval-Augmented Generation) is a safety-first RAG framework that treats **retrieval as a safety-critical action**. The core objective is to **align evidence access**, not only the final generated output.


## 1. Design Goals

**Primary goals**
- **Prevent grounded harm**: avoid generating unsafe answers that are “supported” by retrieved text.
- **Mitigate retrieval-driven jailbreaks**: stop adversarial prompts from forcing retrieval of harmful documents.
- **Auditability**: every decision (retrieve/restrict/no-retrieve) must be justified with structured traces.
- **Modularity**: retriever, judge, search, and generator are swappable under a consistent safety contract.
- **Reproducibility**: deterministic runs (seed), logged traces, evidence provenance, and config snapshots.


## 2. Threat Model (RAG-Specific)

RAI-RAG is designed against:
- **Evidence injection**: user crafts queries to retrieve harmful or policy-violating sources.
- **Grounded unsafe generation**: model produces disallowed instructions while citing retrieved evidence.
- **Query steering / forced prefixes**: prompt includes instructions to bypass safety and retrieve anyway.
- **Distribution shift attacks**: adversarial prompt families unseen in training (role-play, narrative camouflage).

Out of scope:
- Full adversarial control of the document store (can be mitigated via secure corpora and allowlists).


## 3. High-Level Pipeline

**Stage 1 — Prompt Ingestion**
- Normalize input prompt; treat as untrusted.
- Optional: sanitize/strip high-risk PII in logs.

**Stage 2 — Structured Introspection**
- Produce explicit `<Reasoning_step>` trace and `<IR_JSON>`.
- Outputs: intent hypothesis, risk category, severity, ambiguity, retrieval need, retrieval risk, response mode.

**Stage 3 — Retrieval Gate**
- Decide action ∈ `{Retrieve | Restrict | No-Retrieve}`.
- If restricted: enforce allowlists, denylist terms, reduced `top_k`, query rewriting.

**Stage 4 — Retrieval + Evidence Safety Filter**
- Retrieve candidates `D` using BM25/FAISS/hybrid.
- Score each doc with evidence safety scorer `S_e(d|x,IR) ∈ [-1,1]`.
- Prune unsafe docs; return safe bundle `D_safe` + audit metadata.

**Stage 5 — Answer Synthesis**
- Generate response conditioned on `IR` + `D_safe`.
- Output modes:
  - Safe grounded response (preferred)
  - Safe high-level response (when evidence borderline)
  - Refusal with alternatives (when disallowed intent)


## 4. Execution Flow Diagram

```text
(1) Prompt x
   ↓
(2) Introspection → IR_JSON (intent, risk, constraints)
   ↓
(3) Retrieval Gate → action + query plan
   ↓
(4) Retrieve D → Evidence Filter → D_safe + audit
   ↓
(5) Synthesis → final answer (safe / high-level / refusal)

