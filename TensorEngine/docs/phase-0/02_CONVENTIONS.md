# Fase 0.2 — Convenciones matemáticas

Este documento es normativo. Todo backend y toda prueba deberán respetarlo.

## 1. Geometría

- Firma lorentziana mayormente positiva: `(-,+,+,…,+)`.
- Índices latinos abstractos: `a,b,c,… = 0,…,D-1`.
- La conexión es torsión-cero y compatible con la métrica:

  \[
  \Gamma^a{}_{bc}=\Gamma^a{}_{cb},
  \qquad \nabla_a g_{bc}=0.
  \]

- Convención del tensor de Riemann:

  \[
  R^a{}_{bcd}
  =\partial_c\Gamma^a{}_{db}-\partial_d\Gamma^a{}_{cb}
  +\Gamma^a{}_{ce}\Gamma^e{}_{db}
  -\Gamma^a{}_{de}\Gamma^e{}_{cb}.
  \]

- Riemann completamente covariante:

  \[
  R_{abcd}=g_{ae}R^e{}_{bcd}.
  \]

- Ricci y escalar de Ricci:

  \[
  R_{bd}=R^a{}_{bad},
  \qquad R=g^{bd}R_{bd}.
  \]

Esta convención coincide con la usada por la infraestructura coordenada previa
del proyecto en `Simulation/ModelosConAcoplamiento/mc_geometry.py`.

## 2. Simetrías

El Riemann satisface

\[
R_{abcd}=-R_{bacd}=-R_{abdc}=R_{cdab},
\qquad R_{a[bcd]}=0.
\]

El momento `P^{abcd}` es la derivada restringida al espacio de tensores con
estas simetrías y, por tanto, debe heredar las mismas simetrías algebraicas.

Las convenciones de peso son

\[
A_{(ab)}=\frac12(A_{ab}+A_{ba}),
\qquad
A_{[ab]}=\frac12(A_{ab}-A_{ba}).
\]

Para más de dos índices se usa peso `1/n!`.

## 3. Variable métrica de variación

La variable fundamental para presentar la variación es la métrica inversa
`g^{ab}`. En consecuencia,

\[
\delta g_{ab}=-g_{ac}g_{bd}\,\delta g^{cd},
\]

\[
\delta\sqrt{-g}
=-\frac12\sqrt{-g}\,g_{ab}\delta g^{ab}.
\]

La variación de la conexión se escribe inicialmente como

\[
\delta\Gamma^a{}_{bc}
=\frac12g^{ad}
\left(
\nabla_b\delta g_{dc}
+\nabla_c\delta g_{db}
-\nabla_d\delta g_{bc}
\right).
\]

La identidad de Palatini es

\[
\delta R^a{}_{bcd}
=\nabla_c\delta\Gamma^a{}_{db}
-\nabla_d\delta\Gamma^a{}_{cb}.
\]

## 4. Campo escalar

Se define

\[
u_a\equiv\nabla_a\phi=\partial_a\phi.
\]

Como `φ` es escalar,

\[
\delta u_a=\nabla_a\delta\phi.
\]

Al mantener `u_a` covariante como argumento independiente, su variación no
contiene una contribución explícita de la conexión. La dependencia métrica de
objetos como `u^a=g^{ab}u_b` se registra en `M_ab`.

## 5. Acción, ecuaciones y frontera

La variación se normaliza como

\[
\delta S
=\kappa\int_{\mathcal M}d^Dx\sqrt{-g}\,
\left[
E_{ab}\delta g^{ab}
+E_\phi\delta\phi
+\nabla_a\Theta^a
\right].
\]

Las ecuaciones de campo son

\[
E_{ab}=0,
\qquad E_\phi=0.
\]

`E_ab` debe almacenarse en forma manifiestamente simétrica. La corriente
`Θ^a` es el potencial de frontera de la variación; no debe confundirse con una
corriente de Noether ni con una carga superficial.

La ecuación escalar usa la convención

\[
E_\phi=F_\phi-\nabla_aJ^a.
\]

## 6. Igualdad y simplificación

Dos expresiones tensoriales se consideran equivalentes solo después de aplicar
las reglas declaradas para:

- índices mudos;
- simetrías tensoriales;
- compatibilidad métrica;
- identidades de Bianchi habilitadas;
- hipótesis específicas del modelo.

Una expresión que el backend no pueda reducir no se declarará cero. El estado
de una verificación será `passed`, `failed` o `undetermined`.

