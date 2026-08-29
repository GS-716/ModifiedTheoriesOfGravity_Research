# Fase 4.2 — Momentos del lagrangiano

Para un escalar `L(g,R,phi,u)` se calculan, manteniendo independientes los otros
argumentos,

\[
M_{ab}=\frac{\partial L}{\partial g^{ab}},\qquad
P^{abcd}=\frac{\partial L}{\partial R_{abcd}},\qquad
J^a=\frac{\partial L}{\partial u_a},\qquad
F_\phi=\frac{\partial L}{\partial\phi}.
\]

`LagrangianMomenta` almacena estos cuatro objetos con claves estables y permite
serializarlos a JSON.

## Proyecciones

La derivada respecto de la métrica inversa se proyecta sobre la parte simétrica,
por lo que `M_ab` es manifiestamente simétrico.

La derivada respecto de Riemann se proyecta sobre:

- antisimetría en cada par;
- simetría bajo intercambio de pares;
- identidad cíclica de Bianchi.

Así, `P^{abcd}` hereda las simetrías algebraicas completas fijadas en fase 0.
No se introducen factores combinatorios ocultos fuera de esta proyección.

Como referencia, para

\[
L=R=g^{ac}g^{bd}R_{abcd}
\]

el motor obtiene

\[
P^{abcd}=\frac12\left(g^{ac}g^{bd}-g^{ad}g^{bc}\right).
\]
