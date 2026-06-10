## Efficiency Comparison (auto-generated)

| Method | Profile | Class Match | Mean Time | Expr. Proposed | Optim Calls |
|--------|---------|-------------|-----------|----------------|-------------|
| ADCD | mock | 34/36 (94.4%) | 4.8s | 3,225 | 598 |

### Gate Funnel (aggregate)

| Stage | Count |
|-------|------:|
| input_count | 3,225 |
| parse_fail | 0 |
| ast_reject | 0 |
| dim_reject | 337 |
| transcendental_reject | 0 |
| arc_reject | 2,233 |
| coarse_reject | 0 |
| output_count | 655 |
| overall survival | 20.3% |
| PySR | fast | 16/36 (44.4%) | 2.9s | ~0 hall-of-fame | — |
| PySR | fair | 15/36 (41.7%) | 8.8s | ~14 hall-of-fame | — |
