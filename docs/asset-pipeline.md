# Asset Pipeline & Job System Workflow

This document summarizes the design and workflow of a versioned asset pipeline with job provenance, remastering, and monorepo strategies.

---

## 1. Authored Assets

* **authored_asset**: identity of asset (Blend file, texture, etc.)
* **authored_asset_version**: immutable versions of the asset
* **authored_version_links**: doubly-linked list to traverse versions forward/backward
* **locked_by**: tracks which user has it checked out

### Key Points

* Authored assets can be updated with new versions.
* Doubly-linked list enables traversal and optional cold storage cleanup.

---

## 2. Generated Assets & Provenance

* **generated_asset**: output of pipeline jobs (material, pmesh, texture, etc.)
* **asset_provenance**: links each generated asset to the specific authored asset versions it was derived from
* **job_id**: explicit reference to the pipeline code that generated it

### Benefits

* Deterministic builds: exact authored inputs + job version = same outputs
* Reproducibility and auditing: every generated asset traces back to input versions and pipeline
* Supports incremental builds: only assets impacted by changes are rebuilt

---

## 3. Job / Pipeline System

* **job**: represents a pipeline step with code hash
* Jobs are explicit provenance for generated assets
* Changes in job code trigger new generated asset versions

### Workflow

1. Development branch (bleeding edge) for experimentation
2. Release branch/tag for stable pipeline releases
3. Generated assets reference specific job version
4. Remastering uses old authored assets + new pipeline release

---

## 4. Remastering Assets

* Start with original authored asset versions
* Select a new pipeline release
* Rebuild assets
* DAG naturally branches:

```text
Original:
   AuthoredA v3 ─┐
                  ├─> GeneratedOld G1 (pipeline v1.0)
   AuthoredB v2 ─┘

Remaster:
   AuthoredA v3 ─┐
                  ├─> GeneratedNew G2 (pipeline v2.0)
   AuthoredB v2 ─┘
```

* Generated assets reference old authored versions + new job version
* Optional: update authored assets if needed

---

## 5. Monorepo & Forking

* Asset pipeline lives in monorepo
* Trunk = bleeding-edge development, not production
* Release versions = stable, reproducible builds
* Forks allow experimentation or remasters without affecting trunk or releases

### External Consumers

* Websites or services can live outside the monorepo
* Consume only **generated assets**
* Independent CI/CD and branching rules

---

## 6. Summary Mental Model

```text
Monorepo:
    trunk/dev → latest job code
    pipeline-v1.0 → GameA production
    pipeline-v1.1 → GameB production

Fork / Remaster:
    trunk/dev → experimental features
    pipeline-v2.0 → remaster pipeline for GameA

Generated assets DAG:
    Authored assets + Job versions → Generated assets
    Provenance edges maintain deterministic build history
```

* Trunk is free to evolve without impacting old releases
* Remastering and forks integrate seamlessly with explicit provenance
* Versioned jobs + DAG ensure reproducibility and determinism

---

This workflow balances:

* Flexibility for developers (bleeding-edge trunk, forks)
* Stability for production (release pipeline versions, deterministic builds)
* Long-term maintenance (remastering, old game builds, external consumers)

