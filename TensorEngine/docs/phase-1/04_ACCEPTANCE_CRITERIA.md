# Fase 1.4 — Criterios de aceptación

## ModelSpec

- [x] Es independiente de SymPy y xAct.
- [x] Es inmutable y serializable.
- [x] Valida dimensión, parámetros, funciones y convenciones.
- [x] Rechaza símbolos y tensores no declarados.
- [x] Rechaza lagrangianos con índices libres.

## Representación intermedia

- [x] Representa racionales exactos, escalares, tensores, sumas, productos,
  potencias y funciones.
- [x] Registra nombre, varianza y espacio de cada índice.
- [x] Valida contracciones y sumas tensoriales.
- [x] Tiene serialización y deserialización estructuradas.
- [x] Reserva derivadas covariantes para resultados futuros.

## Contratos

- [x] La topología del pipeline está declarada y validada.
- [x] Los resultados conservan backend, operación y duración.
- [x] Las verificaciones admiten `passed`, `failed` y `undetermined`.
- [x] Los fallos no pueden perder su residual o diagnóstico.

## Pruebas mínimas

- [x] Contracción del escalar de Ricci.
- [x] Contracción del término cinético escalar.
- [x] Rechazo de contracciones inválidas.
- [x] Rechazo de índices libres y símbolos no declarados.
- [x] Round-trip JSON de expresiones y modelos.
- [x] Validación de contratos de etapas.

## Próxima tarea recomendada

Implementar el núcleo tensorial abstracto que opera sobre la IR: sustitución
segura, renombrado de índices mudos, simetrización, contracción métrica y
canonización, antes de conectar SymPy o xAct.

