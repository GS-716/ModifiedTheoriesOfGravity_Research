# Fase 3.2 — Operadores e identidades diferenciales

## Operadores

El núcleo proporciona:

- gradiente de un escalar;
- Hessiano covariante;
- divergencia respecto a un índice libre concreto;
- laplaciano `g^{ab}∇_a∇_b`;
- derivada de Lie respecto al vector abstracto `xi^a`;
- conmutador de derivadas covariantes;
- acción algebraica de Riemann sobre cada índice libre.

La derivada de Lie usa, para un tensor arbitrario,

\[
(\mathcal L_\xi T)^{a\cdots}{}_{b\cdots}
=\xi^c\nabla_cT^{a\cdots}{}_{b\cdots}
-T^{c\cdots}{}_{b\cdots}\nabla_c\xi^a
+T^{a\cdots}{}_{c\cdots}\nabla_b\xi^c+\cdots.
\]

## Conmutador

Con la convención de fase 0,

\[
[\nabla_a,\nabla_b]V^c=R^c{}_{dab}V^d,
\qquad
[\nabla_a,\nabla_b]\omega_c=-R^d{}_{cab}\omega_d.
\]

El motor construye independientemente el lado diferencial y la acción de
Riemann. Para escalares, el residual se reduce a cero. Para tensores generales,
el backend estructural conserva un residual `undetermined` hasta disponer de
canonización multitémino especializada.

## Bianchi diferencial

Se construye el residual

\[
\nabla_eR_{abcd}+\nabla_aR_{becd}+\nabla_bR_{eacd}.
\]

El backend estructural no lo descarta ni lo supone cero: devuelve el residual
con estado `undetermined`.

