# Fase 2.1 — Operaciones del núcleo tensorial

La fase 2 añade operaciones matemáticas sobre la IR sin introducir sintaxis de
SymPy ni Wolfram Language.

## Índices

El núcleo implementa:

- recorrido de todas las apariciones de índices;
- renombrado simultáneo;
- renombrado exclusivo de índices libres;
- canonización determinista de índices mudos;
- producto tensorial sin contracciones accidentales;
- producto de Einstein mediante nombres coincidentes y varianza opuesta;
- detección de sustituciones que alteran la firma tensorial.

La canonización de índices mudos es idempotente. Dos expresiones que solo
difieren en los nombres de índices contraídos adquieren la misma representación.

## Álgebra

También se implementa:

- sustitución estructural exacta;
- distribución de productos sobre sumas;
- linealidad de la derivada covariante sobre sumas;
- simetrización y antisimetrización con peso `1/n!`;
- permutación simultánea de índices libres;
- introducción explícita de métricas para subir o bajar índices.

## Contracciones métricas

El backend estructural puede reducir contracciones directas como

\[
g^{ab}u_b=u^a,
\qquad
g^{ab}g_{bc}=\delta^a{}_c,
\qquad
\delta^a{}_a=D.
\]

Esta capacidad actúa sobre factores tensoriales explícitos. No intenta deducir
identidades ocultas dentro de funciones simbólicas u objetos todavía no
expandidos.

