# Compilación y procedencia

La compilación produce un `ModelSpec` ordinario; todas las fases tensoriales
posteriores reciben exactamente la misma IR que un modelo construido por código
o proveniente del catálogo.

El modelo compilado conserva en `metadata`:

- expresión original;
- versión del esquema de fuente;
- fingerprint SHA-256 de toda la especificación declarativa.

Como el fingerprint forma parte del `ModelSpec`, queda incluido también en el
fingerprint de modelo que vincula la evidencia xAct.

La consola ofrece dos flujos:

```powershell
tensor-engine compile source.json model.json
tensor-engine run-source source.json --output-root outputs/run --wolfram
```

El primero permite inspeccionar o versionar la IR compilada. El segundo compila
y ejecuta sin crear un archivo intermedio obligatorio.
