# Curriculum Learning PINN

> **Category**: `pinn/` — Training strategy (progressive difficulty)
> **Problem**: Burgers equation with varying viscosity (Re=10→100→1000)

---

## 1. What Makes This Tutorial Unique?

| Aspect | Existing PINNs | **THIS (Curriculum)** |
|--------|--------------|----------------------|
| **Training** | Fixed parameters | **Progressive difficulty** |
| **Viscosity** | Fixed ν | **ν decreases over stages** |
| **Warm start** | None | **Transfer weights between stages** |
| **Focus** | Physics problem | **Training strategy** |

---

## 2. Concept

```
Stage 1: ν=0.1   (Re=10)    → Easy: smooth solution, learn basic structure
Stage 2: ν=0.01  (Re=100)   → Medium: sharper gradients, warm start
Stage 3: ν=0.001 (Re=1000)  → Hard: near-shock, warm start
```

Weights are transferred between stages (same model, progressively refined).

---

## 3. How to Run

```cmd
cd E:\physicsnemo-tutorials\pinn\curriculum_learning
python curriculum_learning.py
```

Runs both curriculum (3 stages) and fixed (same total epochs) for comparison.

---

## 4. Results

- `curriculum_vs_fixed.png` — Loss comparison (curriculum vs fixed)
- `curriculum_solutions.png` — Solution snapshots at different times
- `curriculum_concept.png` — Strategy explanation
