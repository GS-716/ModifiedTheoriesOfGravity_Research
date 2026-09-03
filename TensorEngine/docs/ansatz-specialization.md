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

`results.json` conserva bajo `projected_results.ansatz_geometry` la carta, la
métrica, el campo escalar, su modo (`generic`, `specialized` o `absent`) y las
hipótesis. Los bundles antiguos, que solo guardaban el nombre del ansatz, siguen
siendo legibles. El reporte LaTeX/PDF escribe al inicio de la sección proyectada
el elemento de línea y el perfil escalar efectivamente usados.
