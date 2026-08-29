# Evidencia externa ligada al cálculo

## Problema resuelto

Los reportes de las fases 5–7 validan fórmulas de referencia. No contienen
información suficiente para demostrar que una evidencia corresponde al modelo
que se está ejecutando. La fase 11 añade dos identidades SHA-256:

- `model_fingerprint`: JSON canónico del `ModelSpec` original;
- `calculation_fingerprint`: modelo, lagrangiano efectivo, momentos,
  Euler–Lagrange y resultado Noether–Wald sometidos a xAct.

Wolfram devuelve ambos valores sin reconstruirlos desde texto y Python exige
que coincidan exactamente con la solicitud. Un reporte `verify_model` con otro
nombre o fingerprint es rechazado antes de incorporarse a la verificación.

Los vínculos se conservan en `verification.json`, `results.json` y
`manifest.json`.
