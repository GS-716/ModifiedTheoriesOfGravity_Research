# Procedencia y reporte

`WolframPhase5Check` conserva `strategy` y `adjudicates`. El
`VerificationReport` registra cada adjudicación como la terna

```text
(internal_check, operation, external_check)
```

La misma lista se copia al `RunManifest`. De este modo el estado final puede
auditarse desde `verification.json` o `manifest.json` sin reconstruir decisiones
desde mensajes de texto.

La evidencia externa sigue incluyendo los dos SHA-256 introducidos en fase 11.
Los artefactos exportados mantienen además sus hashes de archivo y el `run_id`
por contenido. Una adjudicación altera el informe final, pero no los objetos
matemáticos calculados.
