# pypso
Particle Swarm Optimization and its variants implemented in python

# PSO
1. Original PSO
2. Canonical PSO
3. Bare bones PSO
4. Adaptive inertia weight PSO
5. Canonical PSO with von Neumann topology
6. PSO with aging leader and challenger
7. Dynamic multi-swarm PSO
8. Orthogonal learning PSO
9. Extraordinariness PSO
10. Adaptive Search Diversification in PSO
11. Self-adaptive PSO with multiple velocity strategies
12. Relaxation velocity update PSO
13. Diversity enhancing mechanism and neighborhood search strategies PSO
14. Adaptive PSO
15. Fitness-distance-ratio based PSO
16. Comprehensive learning PSO

# Test functions
## Unimodal functions
1. **Sphere**
   $$
      f(x) = \sum_{i = 1}^{D} x_i ^2
   $$
2. **Schaffer's f6** (2D)
   $$
      f(x,y) = 0.5 + \frac{\sin^2{(\sqrt{x^2 + y^2 })- 0.5}}{[1+0.001\cdot (x^2 + y^2)]^2 }
   $$
## Multimodal functions
1. **Ackley**
   $$
      f(\vec{x}) = -a e^{-b\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i ^2 }} - e ^ {\frac{1}{d}\sum_{i=1}^d \cos(cx_i )} + a + e^1
   $$
   Recommended variable values: $a = 20, b = 0.2, c = 2\pi$
   $f(\vec{x}*) = 0$ at $\vec{x}* = (0,\dots , 0)$