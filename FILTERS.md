# MACOG Safety Explorer — Filter Reference

Quick reference for what each control does.

## At a glance

| Filter | Effect | Speed |
|---|---|---|
| Segment length | Loads a whole new base file, HIN recomputes | 30 s – 2 min first time |
| View (HIN / HRN) | Toggles which layer the map shows | Instant |
| Blur size | HIN recomputes (sliding-window redistribution) | ~2–5 s |
| Threshold scale | Reflags segments (no recount) | Instant |
| Mode | Picks which mode's HIN to display | Instant |
| HRN analysis | Picks which of 5 HRN splits to display | Instant |
| Counties | Filters map + tables | Instant |
| Functional class | Filters map + tables | Instant |
| Crash years | HIN recomputes on filtered crash subset | ~2–5 s |

**Rule of thumb:** blur, segment length, and crash years actually re-run the math. Everything else just filters the display or reclassifies existing counts.

---

## Detail

### Segment length (0.10 / 0.25 / 0.50 / 1.0 mi)
The roadway network cut into segments of this length. Each length is a separate precomputed file. Changing this reloads a new base — first swap is slow, later revisits are instant. **HRN is not affected** (HRN uses the raw variable-length roadway events).

### View (HIN / HRN)
Just toggles between the two analyses on the map. No recomputation.

### Blur size (0–5)
How far a crash's influence spreads along the network. `0` = crash count only on its own segment. `2` (default, matches consultant) = spreads to 2 neighbors on each side with linear weighting. Higher = smoother, longer HIN corridors. Triggers a live HIN recompute.

### Threshold scale (0.5 – 2.0)
Multiplies every per-county per-mode HIN threshold. `1.0` = consultant's values exactly. Below 1.0 = stricter (more segments flagged as HIN). Above 1.0 = looser (fewer flagged). Doesn't recount crashes — only compares existing counts against a new cutoff.

### Mode (Ped / Bike / Buggy / Motorcycle / Vehicle)
Which crash-type HIN layer to show on the map. All 5 are computed regardless — this only picks which one is displayed.

### HRN analysis (5 options)
- `fsi_all_elk-stj_inc` — all-crash FSI, Elkhart + St Joseph, incorporated areas
- `fsi_all_kos-mar_inc` — all-crash FSI, Kosciusko + Marshall, incorporated areas
- `fsi_all_reg_uninc` — all-crash FSI, unincorporated regional
- `fsi_vru_reg_inc` — VRU (ped + bike) FSI, all incorporated
- `fsi_vru_reg_uninc` — VRU FSI, all unincorporated

All 5 are precomputed. This selector just picks which one to display.

### Counties (multiselect)
Filters both HIN and HRN display. HIN uses per-county thresholds so this is a clean filter. HRN is technically approximate here — tiers were fit on multi-county groups, so viewing a single county shows a slice of that joint model, not a single-county ranking. For most exploratory work this is fine.

### Functional class (multiselect, 1–7)
Filters both HIN and HRN by roadway class:
- 1 = Interstate, 2 = Other Freeway (**excluded from analysis entirely** at build time)
- 3–4 = Principal / Minor Arterial
- 5–6 = Major / Minor Collector
- 7 = Local

Display filter only — no recompute.

### Crash years (slider)
Restricts which crash records count toward HIN. Re-runs the sliding-window distribution on the year-filtered subset. **Does not affect HRN** — HRN tiers are baked at build time from the 2019–2023 model. To change HRN's year window you'd re-run BuildBase with a different filter.

---

