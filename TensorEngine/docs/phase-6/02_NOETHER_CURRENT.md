# Fase 6.2 — Corriente e identidad de Noether

Para el lagrangiano escalar `L`, la corriente asociada a difeomorfismos se fija
como

\[
J_\xi^a=\Theta_\xi^a-\xi^aL.
\]

La invariancia difeomorfa de la acción y la convención de variación respecto a
`g^{ab}` producen la identidad fuera de capa

\[
\mathcal I_b
=2\nabla^aE_{ab}+E_\phi\nabla_b\phi=0.
\]

El signo relativo del término escalar no es una elección adicional: sigue de
`delta_xi g^{ab}=-2 nabla^{(a}xi^{b)}` y
`delta_xi phi=xi^a nabla_a phi`.

La corriente se descompone como

\[
J_\xi^a
=2E^a{}_b\xi^b+\nabla_bQ_\xi^{ab}.
\]

Por tanto, sobre las ecuaciones métricas,

\[
J_\xi^a\doteq\nabla_bQ_\xi^{ab}.
\]

`NoetherWaldResult` conserva la corriente, el término de restricción, la
divergencia de la carga, el residual de la descomposición y la identidad de
Noether. Un residual que el backend estructural no pueda reducir permanece
`undetermined`; no se convierte silenciosamente en cero.
