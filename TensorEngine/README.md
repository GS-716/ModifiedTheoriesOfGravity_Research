# TensorEngine

Motor tensorial y variacional para teorías covariantes cuyo lagrangiano local
tiene la forma

\[
L=L(g^{ab},R_{abcd},\phi,\nabla_a\phi).
\]

Python será el orquestador del pipeline. El álgebra tensorial abstracta podrá
delegarse a xAct mediante Wolfram Engine, mientras que SymPy se usará para
álgebra escalar, componentes, validaciones y como backend alternativo en las
operaciones que soporte.

## Estado

Las **fases 0 a 14** están definidas. La fase 0 congela el contrato matemático;
la fase 1 incorpora `ModelSpec` y la representación intermedia; la fase 2 añade
el núcleo tensorial estructural; la fase 3 incorpora cálculo diferencial
covariante formal y operadores geométricos; la fase 4 incorpora variaciones
elementales y los cuatro momentos del lagrangiano; la fase 5 aplica Palatini,
separa bulk y frontera y construye las ecuaciones de Euler-Lagrange.
La fase 5 está además validada localmente mediante 19 comprobaciones cruzadas
con Wolfram Engine, xTensor, xPert y xTras.
La fase 6 incorpora variaciones por difeomorfismos, corriente de Noether,
identidad fuera de capa y potencial de carga de Iyer–Wald; sus nueve referencias
externas también están aprobadas con xAct.
La fase 7 incorpora ansatz serializables, geometría coordenada de Levi-Civita y
proyección de las ecuaciones abstractas a componentes mediante SymPy.
La fase 8 agrega todas las verificaciones en un informe versionado con estados
`success`, `partial` y `failed`, y admite evidencia externa xAct/xCoba.
La fase 9 cierra el pipeline con paquetes de corrida reconstruibles, identidad
por contenido, manifiestos SHA-256 y exportación determinista a JSON y LaTeX.
La fase 10 incorpora la ejecución integral mediante una API única, configuración
de ramas, eventos de progreso, archivos JSON y una interfaz de línea de comandos.
La fase 11 liga la evidencia Wolfram/xAct al fingerprint del modelo y del cálculo,
y valida ocho residuales algebraicos genéricos transportados desde la IR.
La fase 12 amplía esa validación a once identidades —incluidas Bianchi,
descomposición de la corriente e identidad diferencial de Noether— y permite
adjudicar resultados internos indeterminados con evidencia xAct explícita,
unánime y ligada al mismo cálculo.
La fase 13 incorpora autoría mediante invariantes de alto nivel, un catálogo de
cinco familias de lagrangianos y campañas que ejecutan y comparan varios modelos
bajo una configuración común sin perder los casos válidos cuando otro falla.
La fase 14 añade un frontend declarativo seguro: expresiones textuales en
`R`, `X`, `phi`, parámetros y funciones declaradas se compilan a la misma IR sin
usar `eval`, conservando un fingerprint de la fuente dentro del modelo.

Documentos normativos:

- [`docs/phase-0/01_SCOPE.md`](docs/phase-0/01_SCOPE.md): dominio matemático y exclusiones.
- [`docs/phase-0/02_CONVENTIONS.md`](docs/phase-0/02_CONVENTIONS.md): signos, índices y variaciones.
- [`docs/phase-0/03_OUTPUT_CONTRACT.md`](docs/phase-0/03_OUTPUT_CONTRACT.md): resultados de una corrida.
- [`docs/phase-0/04_ACCEPTANCE_CRITERIA.md`](docs/phase-0/04_ACCEPTANCE_CRITERIA.md): condiciones para cerrar la fase.
- [`docs/phase-1/01_MODEL_SPEC.md`](docs/phase-1/01_MODEL_SPEC.md): entrada canónica de modelos.
- [`docs/phase-1/02_INTERMEDIATE_REPRESENTATION.md`](docs/phase-1/02_INTERMEDIATE_REPRESENTATION.md): lenguaje tensorial común.
- [`docs/phase-1/03_STAGE_CONTRACTS.md`](docs/phase-1/03_STAGE_CONTRACTS.md): topología y resultados de etapas.
- [`docs/phase-1/04_ACCEPTANCE_CRITERIA.md`](docs/phase-1/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 1.
- [`docs/phase-2/01_CORE_OPERATIONS.md`](docs/phase-2/01_CORE_OPERATIONS.md): operaciones tensoriales estructurales.
- [`docs/phase-2/02_CANONICALIZATION.md`](docs/phase-2/02_CANONICALIZATION.md): normalización y simetrías.
- [`docs/phase-2/03_BACKEND_CONTRACT.md`](docs/phase-2/03_BACKEND_CONTRACT.md): interfaz y capacidades de backends.
- [`docs/phase-2/04_ACCEPTANCE_CRITERIA.md`](docs/phase-2/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 2.
- [`docs/phase-3/01_COVARIANT_CALCULUS.md`](docs/phase-3/01_COVARIANT_CALCULUS.md): reglas de derivación covariante.
- [`docs/phase-3/02_DIFFERENTIAL_OPERATORS.md`](docs/phase-3/02_DIFFERENTIAL_OPERATORS.md): operadores e identidades.
- [`docs/phase-3/03_DIFFERENTIAL_BACKEND.md`](docs/phase-3/03_DIFFERENTIAL_BACKEND.md): integración con backends.
- [`docs/phase-3/04_ACCEPTANCE_CRITERIA.md`](docs/phase-3/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 3.
- [`docs/phase-4/01_ELEMENTARY_VARIATIONS.md`](docs/phase-4/01_ELEMENTARY_VARIATIONS.md): reglas de variación elemental.
- [`docs/phase-4/02_LAGRANGIAN_MOMENTA.md`](docs/phase-4/02_LAGRANGIAN_MOMENTA.md): derivadas parciales y proyecciones.
- [`docs/phase-4/03_VARIATIONAL_BACKEND.md`](docs/phase-4/03_VARIATIONAL_BACKEND.md): regla de cadena e integración de backend.
- [`docs/phase-4/04_ACCEPTANCE_CRITERIA.md`](docs/phase-4/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 4.
- [`docs/phase-5/01_PALATINI.md`](docs/phase-5/01_PALATINI.md): variación geométrica de conexión y curvatura.
- [`docs/phase-5/02_EULER_LAGRANGE.md`](docs/phase-5/02_EULER_LAGRANGE.md): ecuaciones de campo universales.
- [`docs/phase-5/03_BOUNDARY_AND_WOLFRAM.md`](docs/phase-5/03_BOUNDARY_AND_WOLFRAM.md): potencial de frontera y puente xAct.
- [`docs/phase-5/04_ACCEPTANCE_CRITERIA.md`](docs/phase-5/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 5.
- [`docs/phase-6/01_DIFFEO_VARIATIONS.md`](docs/phase-6/01_DIFFEO_VARIATIONS.md): variaciones generadas por difeomorfismos.
- [`docs/phase-6/02_NOETHER_CURRENT.md`](docs/phase-6/02_NOETHER_CURRENT.md): corriente, restricción e identidad fuera de capa.
- [`docs/phase-6/03_WALD_CHARGE_AND_WOLFRAM.md`](docs/phase-6/03_WALD_CHARGE_AND_WOLFRAM.md): carga de Wald y validación xAct.
- [`docs/phase-6/04_ACCEPTANCE_CRITERIA.md`](docs/phase-6/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 6.
- [`docs/phase-7/01_GEOMETRY_ANSATZ.md`](docs/phase-7/01_GEOMETRY_ANSATZ.md): contrato reutilizable de carta y ansatz.
- [`docs/phase-7/02_COORDINATE_GEOMETRY.md`](docs/phase-7/02_COORDINATE_GEOMETRY.md): geometría de Levi-Civita en componentes.
- [`docs/phase-7/03_COMPONENT_PROJECTION_AND_WOLFRAM.md`](docs/phase-7/03_COMPONENT_PROJECTION_AND_WOLFRAM.md): proyección IR y validación xCoba.
- [`docs/phase-7/04_ACCEPTANCE_CRITERIA.md`](docs/phase-7/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 7.
- [`docs/phase-8/01_VERIFICATION_MATRIX.md`](docs/phase-8/01_VERIFICATION_MATRIX.md): matriz integral de controles.
- [`docs/phase-8/02_REPORT_AND_STATUS_POLICY.md`](docs/phase-8/02_REPORT_AND_STATUS_POLICY.md): contrato y estados del informe.
- [`docs/phase-8/03_EXTERNAL_EVIDENCE.md`](docs/phase-8/03_EXTERNAL_EVIDENCE.md): integración segura de xAct/xCoba.
- [`docs/phase-8/04_ACCEPTANCE_CRITERIA.md`](docs/phase-8/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 8.
- [`docs/phase-9/01_RUN_PACKAGE_AND_ID.md`](docs/phase-9/01_RUN_PACKAGE_AND_ID.md): paquete reconstruible e identidad por contenido.
- [`docs/phase-9/02_MANIFEST_AND_INTEGRITY.md`](docs/phase-9/02_MANIFEST_AND_INTEGRITY.md): manifiesto, hashes y escritura atómica.
- [`docs/phase-9/03_JSON_AND_LATEX.md`](docs/phase-9/03_JSON_AND_LATEX.md): fuente JSON y vista de presentación.
- [`docs/phase-9/04_ACCEPTANCE_CRITERIA.md`](docs/phase-9/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 9.
- [`docs/phase-10/01_END_TO_END_ENGINE.md`](docs/phase-10/01_END_TO_END_ENGINE.md): ejecución integral y normalización.
- [`docs/phase-10/02_CONFIGURATION_AND_EVENTS.md`](docs/phase-10/02_CONFIGURATION_AND_EVENTS.md): opciones, estado y observabilidad.
- [`docs/phase-10/03_CLI_AND_FILES.md`](docs/phase-10/03_CLI_AND_FILES.md): entrada JSON e interfaz de comandos.
- [`docs/phase-10/04_ACCEPTANCE_CRITERIA.md`](docs/phase-10/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 10.
- [`docs/phase-11/01_BOUND_EVIDENCE.md`](docs/phase-11/01_BOUND_EVIDENCE.md): fingerprints y evidencia ligada.
- [`docs/phase-11/02_IR_TO_XACT.md`](docs/phase-11/02_IR_TO_XACT.md): transporte seguro de la IR.
- [`docs/phase-11/03_GENERIC_CHECKS_AND_LIMITS.md`](docs/phase-11/03_GENERIC_CHECKS_AND_LIMITS.md): controles genéricos y límites diferenciales.
- [`docs/phase-11/04_ORCHESTRATION_AND_ACCEPTANCE.md`](docs/phase-11/04_ORCHESTRATION_AND_ACCEPTANCE.md): integración y cierre de fase 11.
- [`docs/phase-12/01_DIFFERENTIAL_STRATEGIES.md`](docs/phase-12/01_DIFFERENTIAL_STRATEGIES.md): estrategias algebraica, Bianchi y diferencial.
- [`docs/phase-12/02_ADJUDICATION_POLICY.md`](docs/phase-12/02_ADJUDICATION_POLICY.md): política conservadora de adjudicación.
- [`docs/phase-12/03_PROVENANCE_AND_REPORTING.md`](docs/phase-12/03_PROVENANCE_AND_REPORTING.md): trazabilidad en reportes y manifiestos.
- [`docs/phase-12/04_ACCEPTANCE_CRITERIA.md`](docs/phase-12/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 12.
- [`docs/phase-13/01_AUTHORING_AND_CATALOG.md`](docs/phase-13/01_AUTHORING_AND_CATALOG.md): invariantes y modelos incorporados.
- [`docs/phase-13/02_CAMPAIGN_CONTRACT.md`](docs/phase-13/02_CAMPAIGN_CONTRACT.md): ejecución uniforme y aislamiento.
- [`docs/phase-13/03_CLI_AND_REPRODUCIBILITY.md`](docs/phase-13/03_CLI_AND_REPRODUCIBILITY.md): flujo de consola y artefactos.
- [`docs/phase-13/04_ACCEPTANCE_CRITERIA.md`](docs/phase-13/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 13.
- [`docs/phase-14/01_SOURCE_GRAMMAR.md`](docs/phase-14/01_SOURCE_GRAMMAR.md): gramática declarativa.
- [`docs/phase-14/02_SECURITY_AND_DIAGNOSTICS.md`](docs/phase-14/02_SECURITY_AND_DIAGNOSTICS.md): política de seguridad y errores.
- [`docs/phase-14/03_COMPILATION_AND_PROVENANCE.md`](docs/phase-14/03_COMPILATION_AND_PROVENANCE.md): compilación, CLI y trazabilidad.
- [`docs/phase-14/04_ACCEPTANCE_CRITERIA.md`](docs/phase-14/04_ACCEPTANCE_CRITERIA.md): cierre verificable de fase 14.

## Ejecución integral

```python
from tensor_engine import TensorEngine

run = TensorEngine().run(model, output_root="outputs/runs")
print(run.status, run.package.run_id)
```

Desde una consola, después de instalar el proyecto:

```powershell
tensor-engine validate model.json
tensor-engine run model.json --output-root outputs/runs
tensor-engine catalog list
tensor-engine campaign campaign.json --output-root outputs/campaigns --wolfram
tensor-engine compile lagrangian-source.json model.json
tensor-engine run-source lagrangian-source.json --wolfram
```

## Estructura prevista

```text
TensorEngine/
├── docs/                  Especificación matemática y arquitectura
├── notebooks/             Interfaz de investigación y ejemplos
├── src/tensor_engine/     Orquestador y backend Python/SymPy
├── wolfram/               Paquetes Wolfram Language/xAct
├── tests/                 Pruebas matemáticas y de regresión
└── outputs/               Resultados generados; no son fuente del motor
```

## Principios del proyecto

1. Una convención matemática tiene una única definición normativa.
2. Cada resultado conserva trazabilidad hasta el lagrangiano de entrada.
3. Los backends deben implementar el mismo contrato, no fórmulas incompatibles.
4. Las identidades especiales de un modelo se declaran; no se presuponen.
5. Ninguna simplificación no demostrada puede convertirse silenciosamente en cero.
6. Los notebooks consumen el motor, pero no contienen su lógica esencial.
