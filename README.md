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

# Test functions
## Unimodal functions
1. **Sphere**
   $$
      f(x) = \sum_{i = 1}^{D} x_i ^2
   $$
2. **Schaffer's f6**
   $$
      f(x,y) = 0.5 + \frac{\sin^2{(\sqrt{x^2 + y^2 })- 0.5}}{[1+0.001\cdot (x^2 + y^2)]^2 }
   $$