# Flujo de investigación

Esta carpeta es la interfaz de trabajo interactivo de
[TensorEngine](../TensorEngine/README.md) y
[FieldEquationsSolver](../FieldEquationsSolver/README.md). El usuario declara el
lagrangiano, sus constantes y funciones, la dimensión y, si desea componentes,
un ansatz. El motor se encarga de la derivación, las verificaciones y los
reportes; el solucionador se invoca después y solo cuando se solicita.

## Abrir y ejecutar el notebook

Abre [`01_modified_gravity_workflow.ipynb`](01_modified_gravity_workflow.ipynb)
desde la raíz del repositorio o desde esta carpeta y selecciona un intérprete con
ambos paquetes instalados. Consulta la
[instalación](../README.md#instalación-y-primera-ejecución).

1. Ejecuta la primera celda de configuración. Carga el código local y define
   `ansatz`, `dimension`, `display_policy`, `VALIDAR_XACT` y `ejecutar(...)`.
2. Ejecuta la celda del caso que quieras estudiar. Los casos no dependen de
   haber ejecutado los anteriores, pero sí de la configuración inicial.
3. Abre la carpeta que imprime `Bundle:` para consultar el reporte y los datos.
4. Después de actualizar el motor, reinicia el kernel y vuelve a ejecutar la
   configuración para evitar mezclar objetos de versiones distintas.

La configuración inicial utiliza Draft 4 en tres dimensiones, $f(r)$ arbitraria
y $\phi=\Phi(r,\varphi)$ estacionaria. La validación xAct está desactivada por
defecto; `VALIDAR_XACT=True` la solicita en las llamadas del notebook.

## Casos que contiene

| Celda | Modelo |
|---|---|
| 1 | `R` |
| 2 | `R + alpha*R**2` |
| 3 | `R + alpha*RicciSq` |
| 4 | `R + alpha*RiemannSq` |
| 5 | `R + alpha*RicciUU` |
| 6 | `R + alpha*R*X` |
| 7 | `R` más una contracción de Riemann con cuatro gradientes del mismo escalar |
| Perfil opcional | Ejemplo explícito con $\phi=p\varphi$ |
| Draft 4: casos 0, 1 y 2 | Un caso por celda, con `f_input` y `phi_input` editables |

La contracción cuártica de la celda 7 se anula por las antisimetrías de Riemann
para el producto de cuatro copias del mismo gradiente escalar. Sirve también
para comprobar la simplificación tensorial del modelo.

## Uso rápido con `ejecutar`

Tras la configuración inicial:

~~~python
run = ejecutar("modelo_cuadratico", "R + alpha*RicciSq")
~~~

Para añadir otras constantes simbólicas:

~~~python
run_acoplado = ejecutar(
    "acoplamiento_escalar",
    "R + 2/ell**2 + beta0*RicciUU",
    parametros_extra=("ell", "beta0"),
)
~~~

`ejecutar(nombre, lagrangiano, *, ansatz_usado=None, parametros_extra=())`
es una ayuda del notebook, no otra implementación del motor. Declara `alpha`
automáticamente cuando aparece en el texto y usa `parametros_extra` para los
otros nombres. `ansatz_usado=None` significa utilizar el ansatz global, no
desactivar la proyección. La dimensión se toma de la variable global
`dimension` y debe corresponder al ansatz.

Para declarar funciones, supuestos, normalización o un modelo sin ansatz,
utiliza directamente `LagrangianSourceSpec` y `TensorEngine.run`.

## Declaración completa: parámetros, constantes y funciones

Ejemplo independiente de la función auxiliar, una vez instalado el paquete:

~~~python
from tensor_engine import (
    DimensionSpec, FunctionSpec, LagrangianSourceSpec, ParameterSpec,
    TensorEngine, draft4_circular_ansatz,
)

source = LagrangianSourceSpec(
    name="escalar_no_minimo",
    expression="F(phi)*R - alpha*X - V(phi)",
    dimension=DimensionSpec(3),
    parameters=(ParameterSpec("alpha", assumptions=("positive",)),),
    functions=(FunctionSpec("F", 1), FunctionSpec("V", 1)),
)
model = source.compile()
run = TensorEngine().run(
    model,
    ansatz=draft4_circular_ansatz(),
    output_root="outputs/usuario",
)
~~~

`ParameterSpec("alpha")` declara una constante simbólica, no le asigna un valor.
Para fijar un coeficiente desde la entrada, escribe una fracción exacta, por
ejemplo `"R + (1/10)*R**2"`. Usa `**` para potencias; el frontend rechaza
decimales aproximados y nombres no declarados.

`FunctionSpec("F", 1)` declara una función de un argumento.
`FunctionSpec("K", 2)` permite compilar expresiones como `K(phi,X)`; su proyección
puede quedar simbólica cuando el backend no admite la contracción contenida
en el argumento. Los supuestos documentan el modelo y se aplican cuando la
operación correspondiente los soporta.

Los alias disponibles son `R`, `X`, `RicciUU`, `RicciSq`, `RiemannSq` y `phi`.
`X` significa $g^{ab}\nabla_a\phi\nabla_b\phi$, sin factor $-1/2$.
La [guía de autoría](../docs/frontend-invariants.md) explica además las
contracciones genéricas y `ModelBuilder`.

## Elegir una geometría

El ansatz Draft 4 es

$$
ds^2=-f(r)d\tau^2+\frac{dr^2}{f(r)}+r^2d\varphi^2,
\qquad \phi=\Phi(r,\varphi).
$$

La coordenada temporal sigue en la métrica, pero no en el campo escalar. El
parámetro `p` solo aparece al imponer un perfil que lo contenga.

Para FLRW, crea un modelo con dimensión cuatro y pasa la geometría apropiada:

~~~python
from tensor_engine import spatially_flat_flrw_ansatz

flrw_model = LagrangianSourceSpec(
    name="flrw_einstein",
    expression="R",
    dimension=DimensionSpec(4),
).compile()
run_flrw = TensorEngine().run(
    flrw_model,
    ansatz=spatially_flat_flrw_ansatz(),
    output_root="outputs/flrw",
)
~~~

Aquí el campo genérico es $\phi(t)$ y la métrica contiene $a(t)$.
Para otra geometría, usa `CoordinateChart` y `GeometryAnsatz`. Si solo buscas
las expresiones abstractas, llama a `TensorEngine().run(model)` sin ansatz.

## Editar `f(r)` y el perfil escalar después de derivar

Las celdas de los casos 0, 1 y 2 contienen `f_input` y `phi_input`. Son entradas
editables: puedes reemplazarlas por expresiones escalares de la IR utilizando
`Scalar`, `Number`, `Function` y operaciones aritméticas.

El siguiente patrón conserva la geometría genérica y añade la especialización
en una misma corrida; supone que `model` tiene dimensión tres:

~~~python
from tensor_engine import AnsatzSpecialization, Function, Scalar

geometry = draft4_circular_ansatz()
_, r, varphi = geometry.chart.coordinates
ell, p = Scalar("ell"), Scalar("p")

f_input = 1 + r**2 / ell**2
phi_input = p * varphi
# Alternativas editables para phi_input:
# Function("Phi", (r,))
# Function("Phi", (varphi,))

specialization = AnsatzSpecialization(
    metric_functions={"f": f_input},
    scalar_field=phi_input,
)
run_especializado = TensorEngine().run(
    model,
    ansatz=geometry,
    specialization=specialization,
    output_root="outputs/especializados",
)
~~~

Los valores anteriores ilustran la entrada; no son una solución demostrada del
lagrangiano elegido. El flujo es derivar, proyectar con el ansatz genérico y
evaluar la especialización solicitada. Un perfil temporal se rechaza para
Draft 4. `GeometryAnsatz.specialize_scalar(...)` también permite crear un
ansatz ya especializado, como hace la celda de ejemplo angular.

Los valores iniciales de las tres celdas Draft 4 son:

| Caso | `f_input` | `phi_input` |
|---|---|---|
| 0 | $r^2/\ell^2-\lambda$ | $0$ |
| 1 | $r^2/\ell^2-\lambda-\alpha_1p^2\log(r/r_0)$ | $p\varphi$ |
| 2 | $(r^2/\ell^2-\lambda)/(1+\beta_0p^2\ell^2/r^2)$ | $p\varphi$ |

En Python, `mass` representa el símbolo `Scalar("lambda")`. Cambiar estos valores
solo afecta la geometría especializada de esa corrida. Véase la
[guía de especialización](../docs/ansatz-specialization.md).

## Consultar resultados y reportes

~~~python
print(run.abstract.lagrangian)
print(run.abstract.curvature_momentum)

quantity = run.projected.metric_euler
print(quantity.status.value, quantity.reason)
if quantity.components is not None:
    print(quantity.components.values)

print(run.package.verification.summary)
~~~

`run.abstract` conserva las expresiones covariantes y `run.projected` las
componentes genéricas. `run.specialized` contiene las componentes posteriores
si se utilizó `specialization=...`. Para consultar estas últimas usa, por
ejemplo, `run_especializado.specialized.metric_euler`.

- Las siete celdas de prueba y el perfil opcional guardan sus bundles en
  `ResearchWorkflow/outputs/notebook_cases/`.
- Las tres celdas Draft 4 los guardan en
  `ResearchWorkflow/outputs/draft4_cases/`.
- En llamadas propias a la API, `output_root` es relativo al directorio de
  trabajo salvo que se pase una ruta absoluta.
- `results.json` conserva los datos completos; `verification.json` contiene los
  controles y `manifest.json` permite verificar la integridad de los archivos.
- `report.tex` y, si se dispone de LaTeX, `report.pdf` presentan las expresiones
  abstractas, proyectadas y las especializadas cuando corresponda.

Los reportes con $\Phi(r,\varphi)$ incluyen además dos extras compactos:
$\Phi(r)$ y $\Phi(\varphi)$, únicamente para $L$ y $P^{abcd}$. Reutilizan las
componentes existentes y conservan sus datos adicionales en
`presentation.json`, sin sustituir los resultados de la corrida.

La variable `display_policy` ya configura una presentación conservadora:
factorización, recolección, fracciones e índices canónicos, con
`aggressive=False`. Para activar xAct se necesitan Wolfram Engine activado,
`wolframscript` y xAct instalados. Una cantidad simbólica o una comprobación
indeterminada debe leerse junto con su motivo; no significa que se haya
demostrado una identidad o que el perfil sea una solución.

## Resolver las ecuaciones cuando se solicite

La última celda añade `solveFieldEquations(run)` como operación posterior e
independiente. Usa `RESOLVER=False` para ejecutar únicamente el proceso
tensorial y `True` cuando quieras reducir e intentar resolver las ecuaciones.
Selecciona la corrida en `run_a_resolver`.

El ejemplo usa exactamente `phi=q*varphi` y mantiene `f(r)` por resolver.
Por defecto toma la proyección genérica, aunque la corrida contenga ya un perfil
especializado. `solve=False` solo reduce; `use_specialized=True` selecciona
explícitamente las componentes especializadas existentes.

El [manual de resolución](../FieldEquationsSolver/README.md) describe los estados,
las restricciones de dominio y la verificación. El PDF compacto y los JSON de
resolución se guardan en un bundle separado bajo
`ResearchWorkflow/outputs/field_equations`.
