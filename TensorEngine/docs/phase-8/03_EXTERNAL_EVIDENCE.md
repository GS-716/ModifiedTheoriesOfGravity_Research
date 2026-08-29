# Evidencia externa y reproducibilidad

Los informes tipados de Wolfram/xAct de fases 5–7 pueden incorporarse mediante
`external_reports`. Sus comprobaciones reciben el prefijo
`external.phaseN.*`, por lo que no sobrescriben ni ocultan los controles del
backend estructural.

La evidencia externa complementa el informe, pero no cambia automáticamente el
estado de un residual interno: dos comprobaciones solo se identifican cuando
existe un mapa matemático explícito para el modelo concreto. Esta restricción
evita usar una identidad de referencia como prueba indebida de otro lagrangiano.

El comando

```powershell
python scripts/verify_phase8_reference.py
```

genera `outputs/phase8_reference_verification.json`. El modelo escalar–tensor
general puede producir estado `partial` con el backend estructural; `--strict`
hace que ese estado también se refleje como error del proceso automatizado.
