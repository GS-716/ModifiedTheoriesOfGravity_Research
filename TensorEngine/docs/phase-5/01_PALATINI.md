# Fase 5.1 — Restitución geométrica y Palatini

La variación independiente `delta R_abcd` de fase 4 se restituye como variación
de la curvatura de Levi–Civita.

El motor representa

\[
\delta\Gamma^a{}_{bc}
=\frac12g^{ad}
\left(
\nabla_b\delta g_{dc}
+\nabla_c\delta g_{db}
-\nabla_d\delta g_{bc}
\right)
\]

y la identidad de Palatini

\[
\delta R^a{}_{bcd}
=\nabla_c\delta\Gamma^a{}_{db}
-\nabla_d\delta\Gamma^a{}_{cb}.
\]

`mixed_curvature_variation` puede conservar `delta_Gamma` como tensor formal o
expandirlo en derivadas de `delta g_ab`.

Para el Riemann completamente covariante también se incluye la variación del
índice bajado:

\[
\delta R_{abcd}
=-g_{am}R_{nbcd}\,\delta g^{mn}
+g_{ae}\delta R^e{}_{bcd}.
\]

Esta separación es esencial: el primer término genera la contribución
algebraica de curvatura a `E_ab`; el segundo genera derivadas de `P^{abcd}` y el
potencial de frontera.
