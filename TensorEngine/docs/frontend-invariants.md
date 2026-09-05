# Frontend, invariantes y API avanzada

## Uso cotidiano

Declare el lagrangiano escalar y el ansatz. Los parámetros y funciones son
declaraciones, no instrucciones tensoriales. La dimensión debe coincidir con
la carta. No introduzca la densidad de volumen dentro de L.

~~~python
from tensor_engine import (
    DimensionSpec, LagrangianSourceSpec, ParameterSpec,
    TensorEngine, draft4_circular_ansatz,
)

source = LagrangianSourceSpec(
    name="eqt_case2",
    expression="R + 2/ell**2 + ell**2*beta0*(3*RicciUU - X*R)",
    dimension=DimensionSpec(3),
    parameters=tuple(ParameterSpec(n) for n in ("ell", "beta0", "p")),
    assumptions=("ell != 0", "beta0 != 0"),
)
model = source.compile()
run = TensorEngine().run(
    model, ansatz=draft4_circular_ansatz(), output_root="outputs",
)
~~~

El motor ejecuta las etapas existentes: momentos, variación, integración por
partes, ecuaciones, Noether, componentes, verificaciones y exportación.
Las vistas run.abstract y run.projected siguen siendo los accesos principales.
Los supuestos se conservan; no todos son aplicados automáticamente por SymPy/xAct.

La llamada anterior usa verificaciones Python. Para solicitar evidencia externa,
añada wolfram_bridge=WolframXActBridge(timeout_seconds=300) a run, después de
importar esa clase y tener Wolfram Engine/xAct instalado y activado.
No se presupone validación xAct cuando no fue solicitada. En el notebook esta
opción se configura una vez mediante VALIDAR_XACT.

## Alias centrales

| Alias | Expansión | Constructor |
|---|---|---|
| R | \(g^{ac}g^{bd}R_{abcd}\) | ModelBuilder.ricci_scalar() |
| X | \(g^{ab}u_a u_b\), sin factor \(-1/2\) | ModelBuilder.kinetic_scalar() |
| RicciUU | \(g^{ac}R_{abcd}g^{be}g^{df}u_eu_f\) | ModelBuilder.ricci_uu() |
| RicciSq | \(R_{ab}R^{ab}\) | ModelBuilder.ricci_squared() |
| RiemannSq | \(R_{abcd}R^{abcd}\) | ModelBuilder.riemann_squared() |
| phi | Campo escalar | ModelBuilder.phi |

Aquí \(u_a=\nabla_a\phi\), tratado como argumento independiente al variar L.
Se usa la convención de Riemann ya fijada en fase 0. No existe una nueva cabeza
tensorial RicciUU, RicciSq o RiemannSq ni reglas de variación especiales para
ellos: los alias desaparecen al compilar. Los alias repetidos se expanden una
sola vez por compilación.

Los alias son combinables, no una lista de lagrangianos permitidos:

~~~python
from tensor_engine import FunctionSpec

source = LagrangianSourceSpec(
    "scalar_tensor", "F(phi)*R + K(phi, X) - V(phi)",
    functions=(FunctionSpec("F"), FunctionSpec("K", 2), FunctionSpec("V")),
)
quadratic = LagrangianSourceSpec(
    "quadratic", "R + alpha*R**2", parameters=(ParameterSpec("alpha"),),
)
quadratic_ricci = LagrangianSourceSpec(
    "quadratic_ricci", "R + alpha*RicciSq",
    parameters=(ParameterSpec("alpha"),),
)
quadratic_riemann = LagrangianSourceSpec(
    "quadratic_riemann", "R + alpha*RiemannSq",
    parameters=(ParameterSpec("alpha"),),
)
~~~

Use enteros y fracciones exactas; las potencias se escriben con **, no con ^.
La normalización global sigue admitiendo solo parámetros, dimensión y números.
Los nombres reservados no pueden redeclararse como parámetros o funciones.

## Contracciones textuales genéricas

~~~python
source = LagrangianSourceSpec(
    "generic",
    expression='''contract(
        Riemann("a", "b", "c", "d"),
        metric("a", "c"),
        metric("b", "e"),
        metric("d", "f"),
        gradient("e"),
        gradient("f")
    )''',
)
~~~

Esta expresión es equivalente a RicciUU tras canonizar orden e índices mudos.
Riemann tiene cuatro índices inferiores; metric representa la métrica inversa
con dos superiores y gradient representa \(u_a\). Los índices deben ser cadenas
literales válidas. contract aplica las contracciones de Einstein ya existentes
y exige que no queden índices libres. Conserva el alcance de índices mudos de
factores escalares anidados.

No hay eval ni ejecución de código contenido en la cadena. Se siguen rechazando
atributos, imports, lambdas, listas, subscripts, argumentos con nombre y llamadas
indirectas. Solo estos constructores admiten cadenas de índices.

## Extender el registro sin modificar el frontend

~~~python
from tensor_engine import DEFAULT_INVARIANTS, InvariantSpec

def kinetic_squared(builder):
    return builder.kinetic_scalar()**2

registry = DEFAULT_INVARIANTS.with_invariant(
    InvariantSpec("X2", kinetic_squared, "Cuadrado de X", version="1"),
)
source = LagrangianSourceSpec("extension", "R + X2")
model = source.compile(registry=registry)
~~~

El registro es inmutable: crear una extensión no cambia otras corridas.
Los duplicados se rechazan; no hay reemplazos silenciosos. El constructor recibe
el ModelBuilder configurado con los símbolos del usuario y debe devolver una
IR escalar compatible con ModelSpec. Debe ser determinista y sin efectos
secundarios. Es código Python de confianza, nunca código leído de una fuente JSON.
Para un invariante tensorial nuevo cambie su versión y añada pruebas.

El JSON de la fuente conserva texto y declaraciones, no funciones ejecutables.
Para recompilar una fuente con alias personalizados hay que volver a proporcionar
el registro. El modelo y el bundle ya compilados contienen toda la IR expandida
y se deserializan sin ningún registro adicional.

La metadata source_invariants guarda versión y SHA-256 de la expansión de cada
alias usado. source_fingerprint sigue identificando la fuente textual; el
fingerprint del modelo incluye además la IR expandida y su procedencia. Así,
la evidencia xAct no se confunde entre dos expansiones distintas de un alias.

## API tensorial avanzada

No es obligatorio crear un alias. ModelBuilder y los nodos IR siguen públicos:

~~~python
from tensor_engine import ModelBuilder, ModelSpec

b = ModelBuilder()
ricci_uu = b.contract(
    b.metric("a", "c"), b.riemann("a", "b", "c", "d"),
    b.metric("b", "e"), b.metric("d", "f"),
    b.scalar_gradient("e"), b.scalar_gradient("f"),
)
model = ModelSpec("advanced", b.ricci_scalar() + ricci_uu)
~~~

Los invariantes cuadráticos pueden escribirse con la misma API, sin depender
del registro:

~~~python
ricci_sq = b.contract(
    b.metric("a", "c"), b.riemann("a", "b", "c", "d"),
    b.metric("b", "e"), b.metric("d", "f"), b.metric("g", "h"),
    b.riemann("g", "e", "h", "f"),
)
riemann_sq = b.contract(
    b.metric("a", "e"), b.metric("b", "f"),
    b.metric("c", "g"), b.metric("d", "h"),
    b.riemann("a", "b", "c", "d"),
    b.riemann("e", "f", "g", "h"),
)
assert ricci_sq == b.ricci_squared()
assert riemann_sq == b.riemann_squared()
~~~

Para operaciones no cubiertas por los constructores, use Tensor, Index, mul y el
álgebra de índices existente, respetando el dominio L(g,Riemann,phi,u).
Un ansatz arbitrario continúa declarándose con GeometryAnsatz.

## Responsabilidad de cada capa

| Capa | Responsabilidad |
|---|---|
| Frontend textual y registro | Analizar, resolver alias y construir ModelSpec |
| IR | Conservar nodos tensoriales inmutables y serializables |
| Álgebra de índices | Validar contracciones, simetrías y renombramientos |
| Backend variacional | Construir momentos, variación y ecuaciones desde la IR |
| Backend de componentes | Aplicar el ansatz a las cantidades ya calculadas |
| Wolfram/xAct | Comprobar las identidades disponibles ligadas a la corrida |
| Exportación | Presentar y serializar resultados y limitaciones, sin recalcularlos |

## Límites que esta extensión no modifica

- El Caso-2 original es 3D. Ejecutar su misma fórmula con FLRW 4D es una
  prueba de compatibilidad, no la misma teoría tridimensional.
- `RicciSq` y `RiemannSq` no activan identidades dependientes de la dimensión.
  En particular, la relación tridimensional entre Riemann, Ricci y R solo puede
  usarse si se declara e incorpora como una identidad separada y validada.
- La extensión de [contracciones delta](delta-contractions.md) resuelve las
  identidades en la IR y sus componentes. La antigua ausencia de componentes
  delta ya no bloquea E_ab/E_phi; otras limitaciones conservan el respaldo abstracto.
- Una proyección no nula con más de 2048 componentes potenciales queda sin
  evaluar: en 4D, nabla_nabla_P tiene 4096. Su IR completa se conserva.
- La identidad diferencial de Noether puede quedar indeterminada tanto
  internamente como con la estrategia xAct disponible. No significa prueba
  fallida ni validación total.
- Un fallo de acceso/licencia o ejecución del runtime Wolfram puede detener
  su etapa: no es un fallo de proyección ni se oculta como validación aprobada.
- En PDF se muestran hasta doce componentes no nulas por tensor; JSON conserva
  toda la proyección obtenida. La presencia de un PDF no implica validación total.

Pruebas: tests/test_invariants.py, tests/test_curvature_invariants.py,
tests/test_source.py y tests/test_source_integration.py. Cubren equivalencia de
alias, constructores y contracciones de bajo nivel; proyecciones independientes
en draft4_circular y FLRW; y los contratos JSON, manifiesto y LaTeX.
TENSOR_ENGINE_RUN_WOLFRAM_TESTS=1 activa la validación externa.

## Verificación histórica de la entrega del frontend (2026-08-30)

Esta medición precede a la extensión de contracciones delta. Sus bundles y
conteos se conservan como trazabilidad histórica, no como estado del backend actual.

- Suite completa con Wolfram/xAct habilitado: 262 pruebas aprobadas en 306.98 s.
- Prueba adicional de huella de expansión, añadida después de iniciar esa suite:
  1 aprobada en 1.66 s. Comprueba que un mismo texto con distintas expansiones
  personalizadas no comparte la huella del modelo.
- Se comprobó igualdad de IR entre alias y construcción de bajo nivel, incluida
  la contracción textual, renombramiento de índices mudos y momentos del Caso-2.
- Notebook: cuatro celdas ejecutadas sin errores con la declaración compacta.
- Ambos bundles: ida y vuelta JSON exacta, integridad SHA-256 del manifiesto,
  once resultados abstractos y once registros de proyección, dos secciones
  principales en LaTeX y revisión visual de las doce páginas de cada PDF.

| Corrida | Dimensión | Proyecciones completadas | Verificación |
|---|---:|---:|---|
| draft4_circular | 3 | 9/11 | 55 aprobadas, 0 fallidas, 2 indeterminadas |
| flat_flrw | 4 | 8/11 | 55 aprobadas, 0 fallidas, 2 indeterminadas |

En estas dos corridas falla la etapa compartida de proyección de ecuaciones
por falta de componentes de delta; por eso se conservan tanto E_ab como E_phi
abstractas. Esto no implica que E_phi necesite delta por sí misma. En FLRW
tampoco se proyecta nabla_nabla_P por el límite de componentes indicado arriba.
Las dos comprobaciones indeterminadas son la identidad diferencial de Noether
interna y su comprobación externa; no se presentan como aprobadas.

Bundles generados, con report.pdf, report.tex, results.json y manifest.json:

- outputs/notebook_quickstart/eqt-case2-draft4-8623e7be75c0
- outputs/frontend_invariants/source-integration-flrw-d57fc6569616

Archivos creados o modificados por esta extensión:

- src/tensor_engine/invariants.py (nuevo registro).
- src/tensor_engine/builders.py (constructores reutilizables).
- src/tensor_engine/source.py (compilador textual y procedencia).
- src/tensor_engine/__init__.py (exportación de la API).
- tests/test_invariants.py y tests/test_source_integration.py (nuevos).
- ../ResearchWorkflow/01_modified_gravity_workflow.ipynb y ../ResearchWorkflow/README.md.
- README.md y docs/frontend-invariants.md.
- docs/phase-14/01_SOURCE_GRAMMAR.md.
- docs/phase-14/02_SECURITY_AND_DIAGNOSTICS.md.
- docs/phase-14/03_COMPILATION_AND_PROVENANCE.md.

No se modificaron para esta extensión los backends variacional, de componentes,
Wolfram/xAct ni la implementación del exportador; se reutilizaron sus contratos
y capacidades actuales.

## Verificación de RicciSq y RiemannSq (2026-09-02)

La extensión conserva ambos invariantes como contracciones ordinarias de la IR.
Los modelos `R + alpha*RicciSq` y `R + alpha*RiemannSq` recorren el mismo backend
variacional que cualquier combinación anterior. Los reportes incorporan como
resultados de primera clase `ricci_squared` y `riemann_squared`: hay trece
cantidades en `run.abstract` y `run.projected`, y siete en `run.derived`.

Referencias coordenadas comprobadas:

| Ansatz | \(R_{ab}R^{ab}\) | \(R_{abcd}R^{abcd}\) |
|---|---|---|
| `draft4_circular` | \(\frac12 f''{}^2+\frac{f'f''}{r}+\frac{3f'{}^2}{2r^2}\) | \(f''{}^2+\frac{2f'{}^2}{r^2}\) |
| `flat_flrw` | \(12[(\ddot a/a)^2+(\dot a/a)^2(\ddot a/a)+(\dot a/a)^4]\) | \(12[(\ddot a/a)^2+(\dot a/a)^4]\) |

La suite local terminó con **343 pruebas aprobadas y 8 omitidas**; las omitidas
son integraciones Wolfram opt-in. La prueba viva específica de los invariantes
aprobó, y xAct redujo a cero las nueve identidades genéricas de cada modelo:
`9 passed, 0 failed, 0 undetermined` tanto para RicciSq como para RiemannSq.

El PDF de integración `quadratic_riemann_draft4_xact` contiene 18 páginas, dos
secciones principales y 26 subsecciones. Se revisaron visualmente sus páginas
renderizadas: las expresiones abstractas y proyectadas de ambos invariantes no
presentan recortes ni superposiciones. Su bundle está en
`output/pdf/quadratic-riemann-draft4-xact-4eb9dbfc8732` y conserva la evidencia
xAct ligada al fingerprint de esa corrida.

Compatibilidad: `RunPackage.from_data` sigue leyendo bundles anteriores a esta
extensión. Reconstruye las dos expresiones abstractas a partir de los símbolos
almacenados y marca sus proyecciones nuevas como simbólicas; no inventa
componentes ni evidencia xAct ausentes del archivo original. La lectura produce
una identidad de corrida nueva porque el paquete reconstruido contiene dos
resultados adicionales, pero acepta y verifica la huella histórica del contenido
original.
