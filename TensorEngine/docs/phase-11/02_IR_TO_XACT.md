# Transporte IR a xAct

La solicitud sigue siendo JSON y no contiene código Wolfram evaluable. El lado
Wolfram acepta únicamente nodos enumerados:

- números racionales, escalares, tensores, suma, producto y potencia;
- funciones declaradas y sus derivadas formales;
- derivadas covariantes;
- variaciones formales mediante xPert;
- densidad de volumen como escalar distinguido.

Los nombres deben satisfacer el mismo patrón seguro de la IR. Parámetros,
funciones e índices se declaran dinámicamente en un contexto aislado y la
métrica define su propia derivada de Levi–Civita. El nombre canónico nunca se
usa como código Wolfram: el traductor aplica un mapeo interno inyectivo. En
particular, `_` se codifica como `$u$`, porque el guion bajo es válido en
TensorEngine (`alpha_1`) pero tiene semántica de patrón en Wolfram Language.
Este detalle interno no modifica la IR, `results.json`, los fingerprints ni los
nombres mostrados al usuario.

El decodificador comprueba de forma enumerada tipo, campos obligatorios, rango,
aridad, órdenes de derivación, varianza de la derivada covariante y espacio de
índices. Un nodo ausente, futuro o incompatible se conserva como evidencia
**indeterminada** con un diagnóstico `IR transport rejected: ...`; nunca se
marca como aprobado. Los mensajes emitidos por xAct ya no se interpretan por sí
solos como un fallo de decodificación: después del transporte, una identidad
solo recibe `passed` si la estrategia de xAct reduce su residual exactamente a
cero.

Además del residual textual compatible con versiones anteriores, una falla de
transporte incorpora `transport_diagnostic` en la respuesta Wolfram y
`diagnostic` en los resultados integrados. El objeto contiene:

- `code` y `category` (`node`, `tensor`, `index`, `function`, etc.);
- `path`, ruta exacta desde `residual` hasta el fragmento rechazado;
- `node_type` y `symbol`, cuando existen;
- `reason`;
- `fragment`, copia JSON-segura del nodo, índice o expresión original.

El fragmento completo se conserva en `verification.json` y `results.json`. El
LaTeX/PDF muestra código, ruta, símbolo y motivo sin añadir una tercera sección
principal. Reportes antiguos sin el campo opcional siguen siendo válidos.

El mapa de curvatura es explícito:

```text
R_TE^a_bcd = -R_xAct_cd b^a
```

La cabeza de Riemann se obtiene mediante `Riemann[CD]`; no se construye
concatenando nombres de símbolos.
