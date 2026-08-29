# Transporte IR a xAct

La solicitud sigue siendo JSON y no contiene código Wolfram evaluable. El lado
Wolfram acepta únicamente nodos enumerados:

- números racionales, escalares, tensores, suma, producto y potencia;
- funciones declaradas y sus derivadas formales;
- derivadas covariantes;
- variaciones formales mediante xPert;
- densidad de volumen como escalar distinguido.

Los nombres deben satisfacer el mismo patrón seguro de la IR. Parámetros y
funciones se declaran dinámicamente en un contexto aislado. Los índices se
asocian a símbolos internos y la métrica define su propia derivada de
Levi–Civita.

El mapa de curvatura es explícito:

```text
R_TE^a_bcd = -R_xAct_cd b^a
```

La cabeza de Riemann se obtiene mediante `Riemann[CD]`; no se construye
concatenando nombres de símbolos.
