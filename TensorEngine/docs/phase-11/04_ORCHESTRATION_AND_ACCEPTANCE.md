# Orquestación y criterios de aceptación

## Uso

La API acepta un puente configurado:

```python
run = TensorEngine().run(
    model,
    output_root="outputs/runs",
    wolfram_bridge=WolframXActBridge(timeout_seconds=180),
)
```

La CLI equivalente es:

```powershell
tensor-engine run model.json --wolfram --output-root outputs/runs
```

La etapa opcional `wolfram_model_validation` precede a `verify`; sus controles
se incorporan con el prefijo `external.model.*`.

## Cierre

La fase se acepta cuando:

1. fingerprints estables identifican modelo y cálculo;
2. Python rechaza un eco o reporte ligado a otro cálculo;
3. el transporte no evalúa texto arbitrario;
4. los ocho residuales algebraicos se ejecutan en xAct;
5. la evidencia aparece en el manifiesto final;
6. API y CLI pueden activar la etapa explícitamente;
7. la corrida escalar–tensor en vivo tiene cero fallos externos;
8. las validaciones especializadas previas y la suite local no presentan regresiones.
