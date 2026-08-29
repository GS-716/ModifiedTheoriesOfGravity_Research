# Fase 0.1 — Alcance matemático

## 1. Teoría admitida

El objeto de entrada es una acción local y covariante

\[
S[g,\phi]=\kappa\int_{\mathcal M} d^D x\,\sqrt{-g}\,
L(g^{ab},R_{abcd},\phi,u_a),
\qquad u_a\equiv\nabla_a\phi,
\]

donde `κ` es una normalización global configurable. El motor no incorpora una
normalización gravitacional fija dentro de `L`.

El alcance inicial admite:

- una variedad lorentziana suave de dimensión `D`;
- una métrica no degenerada `g_ab`;
- conexión de Levi–Civita;
- un campo escalar real `φ`;
- dependencia algebraica arbitraria en `g^{ab}` y `R_abcd`;
- dependencia algebraica arbitraria en `φ` y `u_a = ∇_a φ`;
- funciones escalares, productos, potencias y contracciones completas de estos objetos;
- acoplamientos constantes o simbólicos declarados por el modelo.

`D` puede mantenerse simbólica durante el cálculo abstracto. La evaluación por
componentes requiere fijar una dimensión entera.

## 2. Variables independientes de la regla de cadena

Para definir los momentos del lagrangiano se consideran independientes los
cuatro argumentos

\[
g^{ab},\qquad R_{abcd},\qquad \phi,\qquad u_a.
\]

Esta independencia es una herramienta de diferenciación parcial. Durante la
variación geométrica se restablecen

\[
R_{abcd}=R_{abcd}[g],
\qquad
u_a=\nabla_a\phi.
\]

Por ello, la dependencia métrica debida a subir índices o realizar
contracciones pertenece al momento métrico explícito; la dependencia debida a
la curvatura se procesa mediante el momento de curvatura.

## 3. Objetos fundamentales

El motor debe distinguir, sin reutilizar nombres ambiguos,

\[
P^{abcd}\equiv
\left.\frac{\partial L}{\partial R_{abcd}}\right|_{g,\phi,u},
\qquad
M_{ab}\equiv
\left.\frac{\partial L}{\partial g^{ab}}\right|_{R,\phi,u},
\]

\[
J^a\equiv
\left.\frac{\partial L}{\partial u_a}\right|_{g,R,\phi},
\qquad
F_\phi\equiv
\left.\frac{\partial L}{\partial\phi}\right|_{g,R,u}.
\]

La normalización de estas definiciones queda fijada por

\[
\delta L=M_{ab}\delta g^{ab}
+P^{abcd}\delta R_{abcd}
+F_\phi\delta\phi
+J^a\delta u_a.
\]

No se introducen factores combinatorios ocultos en la definición de `P`.

## 4. Fuera del alcance inicial

Quedan excluidos de esta primera versión:

- torsión o no metricidad;
- una conexión independiente de la métrica;
- dependencia en `∇R`, `∇∇R` o derivadas superiores de la curvatura;
- dependencia en `∇_a∇_b φ` o derivadas superiores de `φ`;
- múltiples campos escalares;
- campos vectoriales, espinoriales o formas diferenciales dinámicas;
- métricas degeneradas;
- términos no locales;
- anomalías cuánticas, medida funcional y cuantización;
- contraterminos holográficos o renormalización on-shell;
- condiciones de contorno específicas de una geometría concreta;
- identidades dimensionales especiales aplicadas sin haber fijado `D`.

Estas exclusiones son extensiones futuras, no prohibiciones arquitectónicas.

## 5. Alcance de los términos de frontera

El motor debe extraer el potencial de frontera producido por la variación de la
acción. No se exige en esta versión construir automáticamente el término
adicional que hace bien puesto un problema de contorno particular, como un
término generalizado de Gibbons–Hawking–York.

