# Fase 0.4 — Criterios de aceptación

La fase 0 se considera cerrada cuando se cumplen todos los puntos siguientes.

## Alcance

- [x] Está definida la clase `L(g^{ab},R_abcd,φ,∇_aφ)`.
- [x] Están enumeradas las dependencias admitidas.
- [x] Están enumeradas las extensiones fuera de alcance.
- [x] Está separado el potencial de frontera de los contraterminos de contorno.

## Convenciones

- [x] Está fijada la firma métrica.
- [x] Está fijado el signo del tensor de Riemann.
- [x] Están fijadas las contracciones de Ricci y del escalar de Ricci.
- [x] Está fijada la variable métrica de variación.
- [x] Está fijado el peso de simetrización y antisimetrización.
- [x] Están definidos `P^{abcd}`, `M_ab`, `J^a` y `F_φ`.
- [x] Está fijada la convención para `E_ab`, `E_φ` y `Θ^a`.

## Contrato

- [x] Están definidos los resultados obligatorios de una corrida.
- [x] Están definidas las verificaciones mínimas.
- [x] Está definida la trazabilidad por etapas.
- [x] Está definida la política para resultados no decidibles.

## Condición de transición

La siguiente fase puede comenzar sin cambiar estas decisiones. Cualquier cambio
posterior deberá registrarse como una modificación explícita de la especificación
y deberá activar nuevamente las pruebas de regresión de convenciones.

## Próxima tarea recomendada

Diseñar `ModelSpec` y la representación intermedia independiente del backend,
sin implementar todavía la derivación tensorial completa.

