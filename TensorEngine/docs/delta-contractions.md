# Contracción segura de la identidad de Kronecker

La contracción se realiza en la **IR canónica**, mediante el álgebra estructural
existente, antes de proyectar componentes o enviar expresiones a Wolfram/xAct.
No hay reglas por lagrangiano ni modificaciones de las fórmulas variacionales.
El notebook sigue declarando únicamente el modelo y el ansatz.

## Reglas y garantías

- `delta(a↑, b↓) V(b↑)` sustituye `b↑` por `a↑` y elimina ese delta.
  También funciona con covectores, tensores de cualquier rango y el orden
  inverso de los dos slots de la identidad.
- Un factor puede ser un bloque completo con derivadas covariantes. Se usa su
  frontera de índices libres, no un reemplazo textual de todos sus índices.
  Así, `delta(a↑, b↓) ∇a V(c↑)` da `∇b V(c↑)`; y
  `delta(a↑, b↓) ∇c ∇a V(b↑)` da `∇c ∇b V(b↑)`.
  **No se permutan derivadas ni se traslada un operador entre factores.**
- Las cadenas de identidades se reducen hasta un punto fijo. Las métricas
  inversas se contraen con las reglas métricas existentes; el delta resultante
  pasa por este mismo reductor.
- `delta(a↑, a↓)` es la dimensión declarada de su espacio. Un delta con dos
  índices libres es un tensor identidad: **no es el escalar uno**.
- Se reservan los nombres de destino antes de renombrar índices mudos. Los
  ámbitos internos de funciones, sumas, productos y derivadas no se capturan.
  La firma libre completa, incluidas varianzas y espacios, se comprueba tras
  cada sustitución. Las simetrías siguen perteneciendo al canonizador existente.
  Las simetrías de pares de Riemann también cubren representaciones mixtas:
  se permutan índices completos con sus varianzas, nunca solo sus nombres.
  Así se reconocen las trazas equivalentes que la contracción deja expuestas,
  sin introducir una regla particular para Ricci o un lagrangiano.
  En una traza intrínseca se normaliza cuál de las dos apariciones mudas está
  elevada, usando la simetría de la métrica de contracción. No se cambian las
  varianzas de índices libres ni de contracciones externas al tensor.
- Los ceros siguen siendo ceros. No se divide por parámetros ni se introducen
  hipótesis sobre coordenadas, funciones, métricas o ansatz particulares.

## API y auditoría

```python
from tensor_engine import contract_deltas, delta_count

reduction = contract_deltas(expr, dimension=3, index_space="M")
reduction.expression
reduction.audit.status  # canonical, partial, symbolic
reduction.audit.events

# Para varios espacios, solo se usan dimensiones declaradas explícitamente:
reduction = contract_deltas(expr, dimensions={"M": 3, "N": 2})

# En una corrida normal se aplica automáticamente:
run.delta_contractions  # tuple[DeltaContractionAudit, ...]
```

Cada pasada registra SHA-256 de entrada/salida, número de deltas antes/después
y eventos `substitute`, `trace` o `retained`. Una sustitución registra el tensor
delta, la ruta en la expresión local y los índices fuente/destino completos.
`retained` incluye un motivo concreto. Los nombres son los de la IR de esa
pasada, después de la higiene de índices; no necesariamente los del texto fuente.

`canonical` indica que esa reducción no dejó deltas; no afirma haber demostrado
todas las identidades tensoriales posibles. `partial` significa que se eliminó
alguno y quedó otro; `symbolic`, que siguen explícitos sin reducción segura.

El backend deduplica las pasadas por hashes de entrada/salida. El historial
incluye álgebra intermedia y verificaciones: sumar sus eventos **no equivale** a
contar deltas distintos en las once cantidades físicas finales.

## Serialización, identidad y presentación

- `results.json` añade el campo opcional `delta_contractions` cuando hay
  historial. `RunPackage.from_data` acepta también archivos antiguos sin él,
  sin recanonizar ni modificar los resultados importados.
- `delta_contractions.json` contiene las pasadas y `final_abstract_counts`, el
  conteo independiente en cada una de las once cantidades finales. El manifiesto
  verifica este archivo con el mismo mecanismo SHA-256 que los otros artefactos.
- El historial auxiliar no entra en `semantic_data`/`run_id`, como tampoco las
  duraciones. Sí se protege su integridad mediante el hash de los archivos del
  manifiesto. Una expresión canónica que cambia por una contracción válida
  **sí cambia su hash**; se genera evidencia xAct nueva, no se reutilizan vínculos
  de una IR distinta. Backend estructural: versión 0.9.0.
- `run.abstract` y `run.projected` conservan su estructura. El exportador usa
  esos resultados, no repite derivaciones. La cabecera del reporte resume la
  auditoría sin añadir una tercera sección principal.
  Las etiquetas de componentes se alinean por identidad de índice con los ejes
  registrados en JSON; no se presupone que el orden interno sea a,b,c,d.
  Esto solo corrige la rotulación: no permuta datos almacenados ni sus hashes.
- `DisplayPolicy.canonicalize_indices` solo renombra índices mudos de forma
  higiénica. Ya no contrae métricas/deltas ni llama al backend variacional.
  `presentation.json` conserva la referencia/hash canónico y las operaciones
  de presentación, independientemente del historial de cálculo.

## Componentes y límites

El backend de componentes conoce la identidad del espacio geométrico activo:
diagonal uno, resto cero. Si queda un delta libre legítimo, puede proyectarlo.
Las trazas intrínsecas de un tensor reutilizan la misma suma de Einstein que
los productos; esto permite evaluar las contracciones expuestas por la reducción.
Las derivadas usan la conexión del ansatz proporcionado, sin sustituirlo por
`draft4_circular`, FLRW ni otra geometría predeterminada.

Se conservan explícitos los deltas sin pareja de contracción, con varianzas
incompatibles, entre espacios distintos sin isomorfismo declarado, o con una
traza de dimensión desconocida. La IR ambigua se devuelve intacta por la API
diagnóstica; esto **no** autoriza a introducir modelos con índices inválidos en
el validador normal. La falta de proyección sigue siendo no fatal y deja su
motivo y respaldo abstracto.

No se amplía el presupuesto de componentes: en FLRW 4D, una segunda derivada
no nula de P tiene 4096 componentes potenciales, por encima del límite de 2048.
La identidad diferencial de Noether puede seguir indeterminada. Reducir deltas
no equivale a implementar todas las identidades diferenciales de curvatura.

## Pruebas

`tests/test_delta.py` cubre reglas directas, bloques derivados, orden de
derivadas, cadenas, trazas, índices libres/mudos, captura de nombres, espacios,
ambigüedad, ceros, auditoría y componentes sobre geometrías planas y curvas de
usuario. También comprueba que DisplayPolicy no hace contracciones.

`tests/test_source_integration.py` reutiliza las corridas del Caso-2 en
`draft4_circular` 3D y FLRW 4D: ausencia de deltas en las once expresiones
abstractas, proyección de E_ab/E_phi, ida y vuelta JSON, integridad del
manifiesto y equivalencia de archivos canónicos al cambiar DisplayPolicy.
`TENSOR_ENGINE_RUN_WOLFRAM_TESTS=1` habilita además las comprobaciones xAct.

## Evidencia de esta entrega (2026-08-30)

- Regresión dirigida del álgebra, variación, Euler y campañas: **82 aprobadas**
  (35.01 s), después de resolver las trazas mixtas expuestas por la reducción.
- Suite completa con Wolfram/xAct: **314 aprobadas y 2 fallos de prueba**
  (410.36 s). Los dos fallos eran accesos a nombres inexistentes escritos en
  las nuevas aserciones (`metric_equation`/`scalar_equation`); se corrigieron a
  los accesos existentes `metric_euler`/`scalar_euler`, sin cambiar la API.
- Repetición completa de `tests/test_source_integration.py` con xAct tras esa
  corrección: **12 aprobadas** (216.74 s), incluidas las dos aserciones anteriores.
- Regresión final de `test_delta.py`, `test_presentation.py` y
  `test_exporting.py`: **60 aprobadas** (8.28 s). Incluye la nueva prueba de
  etiquetas con distinto orden de ejes. Estos conteos se solapan; no deben sumarse.

| Corrida | Proyecciones completadas | Verificación de corrida | Deltas finales |
|---|---:|---|---:|
| Caso-2, draft4_circular, 3D | 11/11 | 58 aprobadas, 0 fallidas, 2 indeterminadas | 0 |
| Misma fórmula, flat_flrw, 4D | 10/11 | 58 aprobadas, 0 fallidas, 2 indeterminadas | 0 |

Ambas proyectan E_ab y E_phi. En draft4, E_ab tiene tres componentes no nulas
y E_phi es cero; en FLRW, E_ab tiene cuatro y E_phi una. La fórmula 4D sirve
como prueba de compatibilidad, no identifica la teoría 3D con otra dimensión.
Las dos verificaciones indeterminadas son la identidad diferencial de Noether
interna y su comprobación externa. El reporte distingue la validación
**estructural** de una comprobación independiente de cada cantidad; no se
presentan como validadas cantidades para las que xAct no dispone de una identidad.

En cada corrida hay 158 pasadas distintas registradas y 2591 sustituciones delta
en esas pasadas, incluidas verificaciones. No quedan deltas en las once
cantidades abstractas finales. Los deltas libres/no seguros se conservan y se
comprueban en las pruebas unitarias; no se inventa uno restante en el Caso-2.

Bundles finales (cada uno contiene report.tex, report.pdf, results.json,
verification.json, presentation.json, delta_contractions.json y manifest.json):

- `outputs/delta_contractions/source-integration-draft4-126986cd8a0c`
- `outputs/delta_contractions/source-integration-flrw-c2ff6870cc3b`

Se revisaron visualmente las diez páginas de cada PDF y el LaTeX. Se corrigió
la rotulación de componentes para respetar su orden de ejes; no se observaron
ecuaciones recortadas ni superpuestas. Cada documento mantiene dos secciones y
veintidós cantidades principales. La presentación no tuvo estados fallback:
14 expresiones simplificadas en draft4 y 18 en FLRW; las otras conservaron su
álgebra con formato legible.

La reexportación conserva byte por byte results.json y verification.json.
Se comprobaron la ida y vuelta RunPackage JSON, los hashes de los manifiestos
y la identidad semántica con las corridas de la última integración xAct.
Los reportes anteriores y el lagrangiano del notebook del usuario no se alteraron.

## Archivos de esta extensión

- Álgebra: `src/tensor_engine/delta.py` (nuevo), `indices.py`, `canonical.py`,
  `backends/structural.py` y `components.py`.
- Integración y salida: `src/tensor_engine/engine.py`, `exporting.py`,
  `presentation.py` y `__init__.py`.
- Pruebas: `tests/test_delta.py` (nuevo), `test_source_integration.py`,
  `test_presentation.py` y `test_exporting.py`.
- Documentación: `README.md`, `docs/delta-contractions.md` (nuevo),
  `docs/display-policy.md`, `docs/frontend-invariants.md` y
  `docs/phase-9/01_RUN_PACKAGE_AND_ID.md`, `02_MANIFEST_AND_INTEGRITY.md`,
  `03_JSON_AND_LATEX.md`.

Los módulos de fórmulas variacionales, Euler, Noether, frontend y transporte
Wolfram no requirieron cambios para esta extensión. Se reutilizan sus objetos
y contratos; los cambios previos del workspace se conservaron.
