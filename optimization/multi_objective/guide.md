# Multi-Objective Pareto Optimization

> **Category**: `optimization/` — AI-Based Design Optimization
> **Paradigm**: Multi-objective inverse design via Pareto front exploration
> **Model**: MLP surrogate + weighted sum gradient descent

---

## 1. What Makes This Tutorial Unique?

| Aspect | Differentiable Design (existing) | Multi-Objective (THIS) |
|--------|----------------------------------|----------------------|
| **Objectives** | 1 (match target Cp) | 3 (max Cl, min Cd, max thickness) |
| **Solutions** | 1 optimal | Pareto front (multiple) |
| **Trade-offs** | None | Core focus |
| **Output** | Single design | Set of non-dominated designs |

### Key Difference: Single vs Multiple Objectives
- **Existing**: One objective → one optimal solution, no trade-offs
- **This**: Three competing objectives → Pareto front of non-dominated solutions
- Real engineering always has competing goals (speed vs efficiency, strength vs weight)

---

## 2. Problem: Airfoil Design with 3 Objectives

```
Design variables: 16 control points (airfoil shape)
Objective 1: Maximize Cl (lift coefficient) — aerodynamic performance
Objective 2: Minimize Cd (drag coefficient) — fuel efficiency
Objective 3: Maximize thickness — structural strength
```

These objectives **conflict**:
- Thicker airfoil → more lift but more drag
- Thinner airfoil → less drag but less lift and less strength

---

## 3. Method: Weighted Sum + Surrogate Backprop

### 3.1 Surrogate Model
- MLP: 16 shape points → [Cl, Cd, thickness]
- Trained on 500 random airfoil shapes
- Enables fast gradient computation

### 3.2 Weighted Sum Scalarization
```
L = w1·(-Cl) + w2·(Cd) + w3·(-thickness)
```
- All objectives converted to minimization
- Sweep weights w1, w2, w3 to trace Pareto front
- Each weight combination → one Pareto point

### 3.3 Pareto Optimality
- Solution A **dominates** B if A is ≥ B in ALL objectives and > in at least one
- **Pareto front** = set of non-dominated solutions
- No single "best" — depends on designer's preference

---

## 4. Code Structure

| Section | Description |
|---------|-------------|
| [1] Problem | 3 objectives, 16 design variables |
| [2] Data | 500 random airfoils, simplified aero metrics |
| [3] Surrogate | MLP (16→128→128→128→3) |
| [4] Train Surrogate | 200 epochs |
| [5] Pareto Optimization | 30 weight combinations, 300 steps each |
| [6] Analysis | Non-dominated sorting |
| [7] Visualization | 3D Pareto front, airfoil shapes, trade-offs |
| [8] Summary | Metrics and key observations |

---

## 5. How to Run

```cmd
cd E:\physicsnemo-tutorials\optimization\multi_objective
python multi_objective.py
```

Results saved to `results/`:
- `multi_obj_surrogate_loss.png` — Surrogate training loss
- `multi_obj_pareto_front.png` — 3D and 2D Pareto fronts
- `multi_obj_airfoils.png` — Airfoil shapes along Pareto front
- `multi_obj_tradeoff.png` — Trade-off analysis
- `multi_obj_explanation.png` — Concept comparison

---

## 6. vs. Differentiable Design (Existing Tutorial)

| Feature | Differentiable Design | Multi-Objective |
|---------|----------------------|-----------------|
| **# Objectives** | 1 | 3 |
| **# Solutions** | 1 | Pareto front |
| **Trade-offs** | None | Core concept |
| **Optimization** | Gradient descent | Weighted sum sweep |
| **Real-world** | Simplified | More realistic |

---

## 7. References

- Pareto, "Manual of Political Economy" (1906)
- Marler & Arora, "Survey of multi-objective optimization methods for engineering" (2004)
- King et al., "Airfoil design using surrogate models" (AIAA)
