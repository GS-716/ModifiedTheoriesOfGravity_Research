# CLI y reproducibilidad

El catálogo puede consultarse y exportarse sin escribir código:

```powershell
tensor-engine catalog list --json
tensor-engine catalog export k_essence models/k_essence.json
```

Una campaña JSON se ejecuta con:

```powershell
tensor-engine campaign campaign.json --output-root outputs/campaign --wolfram --strict
```

La raíz contiene un `campaign-report.json` y, cuando la exportación está activa,
un subdirectorio por clave. Cada subdirectorio conserva el bundle ordinario con
resultados, verificación, manifiesto y LaTeX. El reporte superior registra las
opciones comunes y si Wolfram estuvo habilitado; no sustituye los manifiestos de
cada cálculo.
