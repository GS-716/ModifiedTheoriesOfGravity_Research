# Fase 5.4 — Criterios de aceptación

## Palatini

- [x] Variación de la conexión de Levi–Civita.
- [x] Simetría de `delta Gamma^a_bc` en sus índices inferiores.
- [x] Identidad de Palatini para `delta R^a_bcd`.
- [x] Antisimetría en los índices derivativos de `delta R^a_bcd`.
- [x] Variación completa de `R_abcd`, incluido el índice bajado.
- [x] Forma compacta y forma con conexión expandida.

## Bulk y frontera

- [x] Construcción universal de `E_ab`.
- [x] `E_ab` manifiestamente simétrico.
- [x] Construcción de `E_phi=F_phi-nabla_a J^a`.
- [x] Potenciales `Theta_g^a`, `Theta_phi^a` y total.
- [x] Variación completa y variación de densidad serializables.
- [x] Verificación estructural de la integración por partes escalar.

## Referencias y regresión

- [x] Einstein–Hilbert.
- [x] Constante cosmológica.
- [x] Escalar canónico y potencial.
- [x] Serialización de `EulerLagrangeResult`.
- [x] Compatibilidad con las fases 0 a 4.

## Wolfram/xAct

- [x] Detección no destructiva de `wolframscript`.
- [x] Transporte JSON local sin API web.
- [x] Script `ping` con comprobación de xAct.
- [x] Error controlado cuando el runtime no está disponible.
- [x] Versiones separadas de Wolfram Engine, xTensor, xPert y xTras.
- [x] Ejecución real local contra Wolfram Engine/xAct.
- [x] Diecinueve comprobaciones xAct aprobadas, cero fallidas e indeterminadas.
- [x] Palatini y variación de `R_abcd` comparadas independientemente con xPert.
- [x] Bianchi algebraica comprobada con el proyector de Young de xTras.
- [x] Mapa de convenciones TensorEngine-xAct explícito y probado.
- [x] Informe JSON reproducible y prueba de integración opt-in.
- [ ] Traducción completa IR-xAct-IR: siguiente fase del backend externo.

La validación externa complementa las pruebas estructurales de Python. Una
instalación sin runtime sigue produciendo `BackendUnavailableError`; nunca se
simulan resultados de xAct.
