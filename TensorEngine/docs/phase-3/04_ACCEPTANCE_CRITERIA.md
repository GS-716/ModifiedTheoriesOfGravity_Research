# Fase 3.4 — Criterios de aceptación

## Derivación

- [x] Linealidad y regla de Leibniz.
- [x] Regla de la cadena con derivadas funcionales estructuradas.
- [x] Derivación de potencias racionales.
- [x] Parámetros covariantemente constantes.
- [x] Compatibilidad métrica.
- [x] `∇_a phi = u_a`.
- [x] Serialización de derivadas de funciones.

## Operadores

- [x] Gradiente.
- [x] Hessiano simétrico de escalares.
- [x] Divergencia de vectores y covectores.
- [x] Laplaciano escalar.
- [x] Derivada de Lie de escalares y tensores.

## Identidades

- [x] Construcción de ambos lados del conmutador.
- [x] Signo distinto para índices contravariantes y covariantes.
- [x] Conmutador escalar reducido a cero.
- [x] Residual tensorial preservado cuando no puede decidirse.
- [x] Residual de Bianchi diferencial preservado como `undetermined`.

## Backend

- [x] Operadores añadidos al contrato abstracto.
- [x] Contexto diferencial derivable desde `ModelSpec`.
- [x] Capacidades diferenciales publicadas por el backend.
- [x] Compatibilidad con las pruebas de fases anteriores.

## Próxima tarea recomendada

Construir el cálculo variacional elemental: variación de `g_ab`, `g^{ab}`,
`sqrt(-g)`, `R_abcd`, `phi` y `u_a`, junto con la regla de Leibniz variacional y
la proyección de derivadas respecto a Riemann.

