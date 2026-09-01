# Compilación y procedencia

La compilación produce un `ModelSpec` ordinario; todas las fases tensoriales
posteriores reciben exactamente la misma IR que un modelo construido por código
o proveniente del catálogo.

El modelo compilado conserva en `metadata`:

- expresión original;
- versión del esquema de fuente;
- fingerprint SHA-256 de toda la especificación declarativa.
- `source_invariants`: versiones y SHA-256 de las expansiones IR usadas.

El JSON de una fuente no contiene funciones ejecutables. Recompilar un alias
personalizado requiere proporcionar el registro correspondiente; cargar un
`ModelSpec` o `RunPackage` ya expandido no lo requiere. Las expansiones distintas
producen fingerprints de modelo distintos, incluso con el mismo texto fuente.

Como el fingerprint forma parte del `ModelSpec`, queda incluido también en el
fingerprint de modelo que vincula la evidencia xAct.

La consola ofrece dos flujos:

```powershell
tensor-engine compile source.json model.json
tensor-engine run-source source.json --output-root outputs/run --wolfram
```

El primero permite inspeccionar o versionar la IR compilada. El segundo compila
y ejecuta sin crear un archivo intermedio obligatorio.
