# TensorEngine

TensorEngine es el núcleo de cálculo simbólico de
[ModifiedTheoriesOfGravity_Research](../README.md). Recibe un lagrangiano,
construye su representación tensorial, deriva los resultados variacionales y
los proyecta sobre un ansatz opcional. La interfaz principal es Python; los
notebooks y la consola utilizan el mismo motor.

## Alcance matemático

El motor trabaja con

$$
S=\kappa\int d^D x\,\sqrt{-g}\,
L(g^{ab},R_{abcd},\phi,u_a),
\qquad u_a=\nabla_a\phi.
$$

La geometría utiliza conexión de Levi-Civita, firma mayormente positiva y un
único escalar real. Admite dependencia algebraica en Riemann, métricas, el campo
y su primer gradiente. Esto incluye Einstein-Hilbert, extensiones de tipo
$f(R)$, invariantes cuadráticos, acoplamientos escalares no mínimos y
truncamientos EQT compatibles con este dominio.

La dimensión puede ser simbólica en la IR abstracta; para proyectar se necesita
un entero $D\geq2$ que coincida con la carta. No están incluidos como argumentos
del lagrangiano $\nabla R$, $\nabla\nabla\phi$, una conexión independiente ni
campos dinámicos adicionales. Las derivadas de orden superior que surjan al
variar un lagrangiano permitido sí forman parte del resultado formal.

## Arquitectura y responsabilidades

| Capa | Archivos principales en `src/tensor_engine/` | Responsabilidad |
|---|---|---|
| Interfaz de usuario | `source.py`, `cli.py`; carpeta `notebooks/` | Declarar modelos y ejecutar corridas |
| Autoría e invariantes | `builders.py`, `invariants.py`, `model.py` | Resolver alias, construir contracciones y validar parámetros, funciones y dimensión |
| Representación intermedia (IR) | `ir.py`, `serialization.py` | Conservar expresiones tensoriales inmutables y serializables |
| Álgebra tensorial | `indices.py`, `canonical.py`, `transform.py`, `delta.py`, `differential.py` | Gestionar índices, simetrías, derivadas y contracciones seguras |
| Backend variacional | `backends/`, `variational.py`, `palatini.py`, `euler.py` | Obtener momentos, variación, ecuaciones y potencial de frontera |
| Noether y Wald | `noether.py` | Construir corrientes, identidades y potencial de carga cuando está habilitado |
| Geometría y componentes | `components.py` | Construir la geometría coordenada y evaluar la IR con SymPy |
| Cantidades y vistas | `derived.py` | Organizar resultados abstractos, proyectados y especializados |
| Verificación y puente externo | `verification.py`, `wolfram_bridge.py`; carpeta `wolfram/` | Evaluar controles y asociar evidencia xAct al modelo y cálculo exactos |
| Presentación y exportación | `presentation.py`, `exporting.py` | Crear vistas legibles, JSON, manifiestos y LaTeX/PDF |
| Orquestación y campañas | `engine.py`, `stages.py`, `contracts.py`, `catalog.py`, `campaign.py` | Coordinar operaciones, estados, tiempos y conjuntos de modelos |

El backend estructural de Python realiza la derivación abstracta por defecto.
SymPy resuelve el álgebra escalar y las componentes. Wolfram Engine/xAct
interviene como validación externa solicitada explícitamente; no se necesita
Mathematica para cada operación del pipeline Python.

~~~mermaid
flowchart TD
    U["Notebook, API Python o CLI"] --> S["LagrangianSourceSpec: texto y declaraciones"]
    S --> F["Frontend seguro + InvariantRegistry"]
    B["ModelBuilder: API tensorial avanzada"] --> M["ModelSpec + IR canónica"]
    F --> M
    M --> O["TensorEngine: validar, normalizar y orquestar"]
    O --> A["Álgebra de índices y contracción de deltas"]
    A --> V["Backend variacional: M, P, J y F_phi"]
    V --> E["Palatini e integración por partes: E_ab, E_phi y frontera"]
    E --> N["Noether y Wald opcionales"]
    V --> D["Cantidades derivadas y run.abstract"]
    E --> D
    G["GeometryAnsatz: carta, métrica y escalar"] --> C["Backend de componentes SymPy"]
    D --> C
    C --> P["run.projected"]
    P --> SP["Especialización opcional del ansatz"]
    I["AnsatzSpecialization: funciones y perfil del usuario"] --> SP
    D --> SP
    SP --> Z["run.specialized"]
    D --> Q["Verificación interna y residuales"]
    N --> Q
    P --> Q
    Q --> W["Puente JSON + wolframscript local"]
    W --> X["Wolfram Engine / xAct, si se solicita"]
    X --> EV["Evidencia ligada por fingerprints"]
    D --> OUT["RunPackage y exportación"]
    P --> OUT
    Z --> OUT
    Q --> OUT
    EV --> OUT
    OUT --> JSON["results.json + verification.json + manifiesto"]
    OUT --> DISP["DisplayPolicy + presentation.json"]
    DISP --> TEX["report.tex y PDF si hay compilador"]
~~~

El diagrama muestra responsabilidades y dependencias de datos. La representación
de presentación no se utiliza como entrada de las ecuaciones ni de la validación.

## Instalación y primera ejecución

Requisitos: Python 3.11 o posterior. SymPy es una dependencia del paquete;
Jupyter es necesario si se trabaja con notebooks. Desde esta carpeta:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" jupyterlab ipykernel
.\.venv\Scripts\python.exe -m jupyter lab
~~~

Para trabajar sin notebooks basta instalar `python -m pip install -e .` en el
entorno elegido. Selecciona ese mismo intérprete en tu editor.

Este ejemplo declara el Caso-2 con la geometría genérica del Draft 4:

~~~python
from tensor_engine import (
    DimensionSpec, LagrangianSourceSpec, ParameterSpec,
    TensorEngine, draft4_circular_ansatz,
)

source = LagrangianSourceSpec(
    name="eqt_case2",
    expression="R + 2/ell**2 + ell**2*beta0*(3*RicciUU - X*R)",
    dimension=DimensionSpec(3),
    parameters=(
        ParameterSpec("ell", assumptions=("positive",)),
        ParameterSpec("beta0"),
    ),
)
model = source.compile()
run = TensorEngine().run(
    model,
    ansatz=draft4_circular_ansatz(),
    output_root="outputs/runs",
)
print(run.status.value)
print(run.export_bundle.output_directory)
~~~

Introduce el escalar `L` sin el elemento de volumen. `normalization` permite
declarar un prefactor global por separado; el motor lo incorpora a la
normalización del cálculo. Para un cálculo exclusivamente abstracto, omite
`ansatz`. La carpeta de salida se resuelve respecto del directorio de ejecución.

## Escribir lagrangianos

| Alias textual | Significado | Constructor equivalente |
|---|---|---|
| `R` | Escalar de Ricci | `ModelBuilder.ricci_scalar()` |
| `X` | $g^{ab}u_a u_b$, sin factor $-1/2$ | `ModelBuilder.kinetic_scalar()` |
| `RicciUU` | $R_{ab}u^a u^b$ | `ModelBuilder.ricci_uu()` |
| `RicciSq` | $R_{ab}R^{ab}$ | `ModelBuilder.ricci_squared()` |
| `RiemannSq` | $R_{abcd}R^{abcd}$ | `ModelBuilder.riemann_squared()` |
| `phi` | Campo escalar | `ModelBuilder.phi` |

Los alias se expanden a operaciones sobre la misma IR; no constituyen una lista
cerrada de teorías. Declara cada constante con `ParameterSpec` y cada función
simbólica con `FunctionSpec`. Por ejemplo:

~~~python
from tensor_engine import FunctionSpec

scalar_source = LagrangianSourceSpec(
    name="scalar_tensor",
    expression="F(phi)*R + K(phi, X) - V(phi)",
    dimension=DimensionSpec(3),
    functions=(FunctionSpec("F", 1), FunctionSpec("K", 2), FunctionSpec("V", 1)),
)
scalar_model = scalar_source.compile()
~~~

Esta declaración es válida en el frontend y en la IR. Su evaluación completa
en componentes depende del soporte del backend; véanse las limitaciones abajo.
Usa `**` para potencias y fracciones exactas como `1/2`. El frontend no ejecuta
código Python contenido en el texto.

Para contracciones no cubiertas por alias existen `contract`, `Riemann`,
`metric` y `gradient`, además de la API `ModelBuilder`. El registro admite
nuevos invariantes construidos con esas operaciones. La
[guía de autoría](docs/frontend-invariants.md) describe ambas vías.

## Ansatz y especialización

`draft4_circular_ansatz()` proporciona, en tres dimensiones,

$$
ds^2=-f(r)d\tau^2+\frac{dr^2}{f(r)}+r^2d\varphi^2,
\qquad \phi=\Phi(r,\varphi).
$$

$f(r)$ es arbitraria y el campo es estacionario. Los perfiles que se impongan
sobre este ansatz no deben reintroducir dependencia en $\tau$.
`spatially_flat_flrw_ansatz()` proporciona FLRW plano en cuatro dimensiones,
con $a(t)$ y $\phi(t)$. Una geometría propia se declara mediante
`CoordinateChart` y `GeometryAnsatz`.

Para conservar en una misma corrida la vista genérica y una especialización
posterior, utiliza `AnsatzSpecialization`:

~~~python
from tensor_engine import AnsatzSpecialization, Scalar

geometry = draft4_circular_ansatz()
_, r, varphi = geometry.chart.coordinates
ell, p = Scalar("ell"), Scalar("p")

specialization = AnsatzSpecialization(
    metric_functions={"f": 1 + r**2 / ell**2},
    scalar_field=p * varphi,
)
specialized_run = TensorEngine().run(
    model,
    ansatz=geometry,
    specialization=specialization,
    output_root="outputs/specialized",
)
~~~

Aquí `p` pertenece al perfil coordenado; no aparece en el lagrangiano abstracto
del ejemplo. Los valores propuestos son entradas editables, no una afirmación
de que resuelvan las ecuaciones de ese modelo. Para contrastar una solución,
inspecciona los residuales de `run.specialized`. La
[guía de especialización](docs/ansatz-specialization.md) desarrolla este flujo.

## Resultados y archivos

Los cuatro momentos se definen tratando los argumentos del lagrangiano como
independientes durante la diferenciación parcial:

$$
M_{ab}=\frac{\partial L}{\partial g^{ab}},\qquad
P^{abcd}=\frac{\partial L}{\partial R_{abcd}},\qquad
J^a=\frac{\partial L}{\partial u_a},\qquad
F_\phi=\frac{\partial L}{\partial\phi}.
$$

La variación geométrica restablece $R_{abcd}=R_{abcd}[g]$ y
$u_a=\nabla_a\phi$ para construir $E_{ab}$ y $E_\phi$, incluyendo sus términos
de derivadas y frontera. Si se declara una normalización global, los resultados
de la corrida corresponden al lagrangiano normalizado por el motor.

La corrida organiza $L$, $M_{ab}$, $P^{abcd}$, $J^a$, $F_\phi$, $E_{ab}$,
$E_\phi$, el escalar de Ricci, Ricci al cuadrado, Riemann, Riemann al cuadrado,
$\nabla P$ y $\nabla\nabla P$. También conserva la contribución
de $\nabla\nabla P$ a la ecuación métrica, la variación y los términos de frontera.

| Acceso Python | Uso |
|---|---|
| `run.abstract` | Expresiones tensoriales anteriores a sustituir el ansatz |
| `run.projected` | Componentes en la geometría genérica, con estado y motivo por cantidad |
| `run.specialized` | Vista posterior, presente si se solicita una especialización |
| `run.derived` | Cantidades intermedias y término métrico de derivadas de $P$ |
| `run.package` | Modelo, resultados, verificaciones y procedencia de la corrida |
| `run.stages` | Estado y duración de las operaciones |
| `run.export_bundle` | Ubicación y diagnóstico de los archivos exportados |

El bundle incluye `results.json`, `verification.json`, `manifest.json`,
`presentation.json`, `delta_contractions.json` y `report.tex`. Produce
`report.pdf` cuando encuentra un compilador LaTeX funcional, como `pdflatex` o
`xelatex`. La falta de compilador no impide conservar el JSON y el archivo TeX.

La presentación se organiza en resultados abstractos y proyectados, y añade
una sección especializada cuando corresponde. Los extras para $\Phi(r)$ y
$\Phi(\varphi)$ muestran solo $L$ y $P^{abcd}$ a partir de la proyección
estacionaria; se guardan en `presentation.json` sin reemplazar los resultados
generales. `DisplayPolicy` controla la legibilidad del reporte y mantiene
separada la expresión canónica de su presentación.

## Verificación y límites de interpretación

Para activar la validación externa, instala y activa Wolfram Engine, deja
`wolframscript` accesible y coloca xAct en una ruta de paquetes del kernel.
Python inicia el kernel local e intercambia JSON; no requiere una API web ni
ejecutar manualmente un notebook de Mathematica.

~~~python
from tensor_engine import WolframXActBridge

bridge = WolframXActBridge(timeout_seconds=300)
print(bridge.ping())
validated_run = TensorEngine().run(model, wolfram_bridge=bridge)
~~~

La evidencia xAct se acepta para el modelo y cálculo cuyos fingerprints
coincidan. Los controles distinguen `passed`, `failed` y `undetermined`.
Una proyección completada no implica una validación xAct independiente de cada
componente, y el estado global no sustituye la inspección de cada cantidad.

Limitaciones que deben tenerse presentes:

- El backend puede conservar cantidades simbólicas por complejidad o por nodos
  no evaluables. Por ejemplo, algunas funciones como `K(phi,X)` con contracciones
  tensoriales dentro de sus argumentos no tienen proyección directa disponible.
- Los supuestos se registran, pero cada backend aplica únicamente los que sabe
  interpretar; no debe inferirse una cancelación o identidad no demostrada.
- No se aplican automáticamente identidades especiales de tres dimensiones.
- Se calcula el potencial de frontera variacional, no todos los términos
  adicionales que requiere un problema de contorno particular.
- El motor deriva y evalúa expresiones; no es un solucionador general de las
  ecuaciones de campo.

## Carpetas, consola y documentación

- [`notebooks/`](notebooks/README.md): uso interactivo y ejemplos.
- [`src/tensor_engine/`](src/tensor_engine/): implementación del motor.
- [`wolfram/`](wolfram/README.md): puente y paquetes de validación.
- [`tests/`](tests/README.md): pruebas matemáticas, serialización e integración.
- [`docs/`](docs/): contratos matemáticos, arquitectura y guías.
- [`scripts/`](scripts/): utilidades reproducibles y referencias.
- `outputs/` y `output/`: bundles y reportes generados.

Después de instalar el paquete, también puedes usar:

~~~powershell
tensor-engine compile lagrangian-source.json model.json
tensor-engine run-source lagrangian-source.json --output-root outputs/runs
tensor-engine run model.json --ansatz ansatz.json --output-root outputs/runs --wolfram
tensor-engine catalog list
tensor-engine campaign campaign.json --output-root outputs/campaigns
~~~

Consulta el [alcance matemático](docs/phase-0/01_SCOPE.md), las
[convenciones](docs/phase-0/02_CONVENTIONS.md), la
[guía del frontend](docs/frontend-invariants.md), la
[política de presentación](docs/display-policy.md) y las
[contracciones de Kronecker](docs/delta-contractions.md).
