# Ansatz geométricos y especialización posterior

`draft4_circular_ansatz()` representa por defecto la geometría

\[
ds^2=-f(r)d\tau^2+\frac{dr^2}{f(r)}+r^2d\varphi^2,
\qquad \phi=\Phi(\tau,r,\varphi),
\]

con `f(r)` y `Phi(tau,r,varphi)` arbitrarias. El símbolo `p` no forma parte del
ansatz base. Por ello `gradient("a")`, sus versiones con el índice elevado y
`X` se proyectan en términos de las derivadas parciales de `Phi`.

La derivación covariante del modelo nunca depende del ansatz. La geometría entra
después, en el backend de componentes. Un perfil escalar se impone de forma
explícita sobre una copia inmutable del ansatz:

```python
from tensor_engine import (
    draft4_angular_scalar_profile,
    draft4_circular_ansatz,
)

draft4 = draft4_circular_ansatz()
draft4_p_varphi = draft4.specialize_scalar(
    draft4_angular_scalar_profile("p"),
    assumptions=("phi=p*varphi",),
)
```

Solo una corrida que reciba `draft4_p_varphi` contiene `p` y el gradiente
`(0, 0, p)`. La corrida con `draft4` conserva las derivadas de `Phi`.

Para evitar una explosión algebraica, el backend conserva simbólicamente una
expresión con `Phi` multivariable cuando contiene más de 24 nodos de derivadas
covariantes. El estado se registra como limitación del backend y la corrida no
falla; las cantidades menos costosas siguen proyectándose. El límite está en
`SympyComponentBackend.generic_scalar_derivative_budget` y puede elevarse de
forma explícita para una sesión con recursos suficientes. Un perfil especializado
no activa este resguardo.

## Sustitución posterior de soluciones

`GeometryAnsatz.specialize()` permite sustituir simultáneamente funciones de la
métrica y el campo escalar, sin tocar la IR abstracta del lagrangiano:

```python
from tensor_engine import Function, Scalar

tau, r, varphi = draft4.chart.coordinates
f = Function("f", (r,))
Phi = Function("Phi", (tau, r, varphi))

solution_ansatz = draft4.specialize(
    {
        f: f_solution,       # expresión IR escalar definida por el usuario
        Phi: phi_solution,   # expresión IR escalar definida por el usuario
    },
    name="draft4_solution",
    assumptions=("hipótesis de la solución",),
)
```

Las sustituciones son estructurales y exigen expresiones escalares compatibles.
La secuencia queda separada en: derivación abstracta, proyección con la métrica,
especialización escalar opcional y evaluación con funciones solución.

## Especialización integrada en una corrida

`AnsatzSpecialization` conserva esa secuencia dentro de `TensorEngine.run` y
acepta expresiones escalares de la IR elegidas por el usuario:

```python
from tensor_engine import AnsatzSpecialization, Scalar, TensorEngine

draft4 = draft4_circular_ansatz()
tau, r, varphi = draft4.chart.coordinates
ell, p, mass = Scalar("ell"), Scalar("p"), Scalar("lambda")

specialization = AnsatzSpecialization(
    metric_functions={"f": r**2 / ell**2 - mass},
    scalar_field=p * varphi,
)
run = TensorEngine().run(
    model,
    ansatz=draft4,
    specialization=specialization,
    output_root="outputs",
)
```

La API no contiene soluciones predeterminadas. `f(r)` y `phi` proceden siempre
del objeto entregado por el usuario. La salida conserva tres niveles:

- `run.abstract`: derivación covariante, independiente del ansatz;
- `run.projected`: proyección sobre el Draft 4 genérico con `f(r)` y `Phi`;
- `run.specialized`: componentes obtenidas tras aplicar la especialización.

Cuando existe `run.specialized`, `results.json` almacena la especialización, la
geometría resultante y las trece cantidades completas. El informe añade al final
`Resultados especializados mediante el ansatz`. La validación xAct continúa
ligada a la teoría covariante: el informe no presenta la sustitución coordenada
como si fuera una validación xAct independiente.

El notebook de inicio incluye una celda separada para cada Caso 0, 1 y 2 del
Draft 4. En cada una, `f_input` y `phi_input` son las únicas entradas que hay que
editar para ensayar otra solución.

`results.json` conserva bajo `projected_results.ansatz_geometry` la carta, la
métrica, el campo escalar, su modo (`generic`, `specialized` o `absent`) y las
hipótesis. Los bundles antiguos, que solo guardaban el nombre del ansatz, siguen
siendo legibles. El reporte LaTeX/PDF escribe al inicio de la sección proyectada
el elemento de línea y el perfil escalar efectivamente usados.
