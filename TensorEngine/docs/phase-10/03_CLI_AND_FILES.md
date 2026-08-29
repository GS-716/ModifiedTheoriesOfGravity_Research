# CLI y archivos de entrada

## Archivos

`save_model` y `load_model` escriben y reconstruyen `ModelSpec` como JSON.
`save_ansatz` y `load_ansatz` hacen lo mismo para `GeometryAnsatz`. Las
escrituras son atómicas y no analizan expresiones desde texto libre: reconstruyen
los nodos tipados de la IR.

## Comandos

Con el proyecto instalado (`python -m pip install -e .`):

```powershell
tensor-engine validate model.json
tensor-engine run model.json --output-root outputs/runs
tensor-engine run model.json --ansatz flrw.json --output-root outputs/runs
tensor-engine run model.json --no-noether --no-export --json
tensor-engine run model.json --wolfram --output-root outputs/runs
```

También puede usarse `python -m tensor_engine` con los mismos argumentos.
`--strict` devuelve código 2 para una corrida parcial; un fallo devuelve 1.

## Límite de Wolfram

Desde la fase 11, `--wolfram` genera evidencia nueva ligada al fingerprint del
modelo y del cálculo. Los reportes genéricos proporcionados por la API también
se rechazan si esos fingerprints no coinciden. Los reportes históricos de
referencia de fases 5–7 conservan su carácter especializado.
