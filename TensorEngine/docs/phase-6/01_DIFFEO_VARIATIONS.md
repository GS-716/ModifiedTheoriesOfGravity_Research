# Fase 6.1 — Variaciones por difeomorfismos

La fase 6 usa un campo vectorial arbitrario `xi^a` como generador infinitesimal.
Con la variable métrica inversa adoptada en fase 0,

\[
\delta_\xi g^{ab}=\pounds_\xi g^{ab}
=-2\nabla^{(a}\xi^{b)},
\qquad
\delta_\xi\phi=\pounds_\xi\phi
=\xi^a\nabla_a\phi.
\]

`DiffeomorphismVariation` conserva ambas expresiones como IR serializable. La
variación métrica se construye de forma manifiestamente simétrica y la escalar
usa `u_a=nabla_a phi`, sin introducir coordenadas.

Estas variaciones se evalúan en el potencial de frontera obtenido en fase 5:

\[
\Theta_\xi^a
=\Theta_g^a[\delta_\xi g]
+\Theta_\phi^a[\delta_\xi\phi].
\]

No se presupone que `xi^a` sea Killing ni que los campos satisfagan sus
ecuaciones. Toda la construcción se mantiene fuera de capa.
