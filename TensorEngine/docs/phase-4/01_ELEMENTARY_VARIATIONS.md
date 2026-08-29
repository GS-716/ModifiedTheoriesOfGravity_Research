# Fase 4.1 — Variaciones elementales

La fase 4 introduce dos nodos calculados en la representación intermedia:

- `Variation(expr)` representa `delta(expr)` y conserva sus índices libres;
- `VolumeElement(g)` representa la densidad `sqrt(-g)` sin tratarla como un
  escalar ordinario.

La variación algebraica aplica linealidad, regla de Leibniz, regla de la cadena
y derivación de potencias. Para un exponente simbólico general aparece la
función formal `Log`; no se imponen automáticamente hipótesis de signo o rama.

Los parámetros declarados y la dimensión simbólica tienen variación cero.

## Relaciones geométricas

La variable métrica fundamental sigue siendo `g^{ab}`. El motor construye

\[
\delta g_{ab}=-g_{ac}g_{bd}\,\delta g^{cd},
\qquad
\delta\sqrt{-g}=-\frac12\sqrt{-g}\,g_{ab}\delta g^{ab}.
\]

Para el campo escalar se distingue entre la variación independiente `delta u_a`
de la regla de cadena y la restitución geométrica

\[
\delta u_a=\nabla_a\delta\phi.
\]

`delta R_abcd` permanece como variación independiente en esta fase. Su
expansión mediante la identidad de Palatini pertenece a la etapa de separación
entre bulk y frontera.
