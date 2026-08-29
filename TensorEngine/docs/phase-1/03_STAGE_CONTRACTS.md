# Fase 1.3 — Contratos de etapas

El pipeline inicial queda declarado como datos. En esta fase se valida su
topología, pero todavía no se ejecutan las operaciones matemáticas. Sus entradas
externas son `model_spec` y, para la rama opcional de componentes,
`geometry_ansatz`.

```text
validate_model
  → normalize_lagrangian
  → derive_momenta
  → raw_variation
  → integrate_by_parts
  ├→ noether                 [opcional]
  ├→ components              [opcional; recibe además GeometryAnsatz]
  ├→ wolfram_model_validation [opcional; evidencia ligada por fingerprint]
  └→ verify
       → export              [opcional]
```

Cada `StageSpec` declara:

- clave única;
- entradas requeridas;
- salidas producidas;
- carácter obligatorio u opcional;
- descripción de la operación.

Cada `StageResult` registra:

- estado `success`, `failed` o `partial`;
- backend responsable;
- operación;
- claves de entrada;
- expresiones de salida;
- artefactos estructurados no tensoriales, como manifiestos o reportes;
- verificaciones;
- diagnósticos;
- duración.

Las expresiones equivalentes se distinguen mediante las formas `raw`,
`canonical` y `model_reduced`.

Una verificación fallida o indeterminada conserva obligatoriamente una expresión
residual. Una etapa fallida conserva obligatoriamente un diagnóstico.

Los artefactos estructurados solo admiten datos JSON-compatibles. Un resultado
marcado como exitoso debe registrar todas las entradas requeridas y producir
todas las salidas declaradas por su `StageSpec`.

`StageSpec`, `StageResult`, expresiones, verificaciones, diagnósticos y artefactos
pueden serializarse y reconstruirse sin depender del proceso que los produjo.
