# Pruebas

Esta carpeta contiene pruebas unitarias, identidades matemáticas y casos de
regresión. Las pruebas acumuladas cubren contratos, IR, modelos, higiene de
índices, canonización, cálculo diferencial covariante y cálculo variacional.
La fase 5 añade referencias de Palatini, Euler-Lagrange, frontera y transporte
local hacia Wolfram/xAct. La integración externa puede habilitarse con
`TENSOR_ENGINE_RUN_WOLFRAM_TESTS=1`; comprueba 19 identidades usando xTensor,
xPert y xTras, mientras la suite normal sigue siendo independiente del runtime.

La fase 6 añade regresiones para difeomorfismos, corriente fuera de capa,
potencial de Komar/Wald, identidad de Noether y serialización. La operación
externa `verify_phase6` añade nueve comprobaciones xAct bajo la misma bandera.

La fase 7 añade contratos de ansatz, traducción escalar IR–SymPy, geometría
FLRW y proyección de ecuaciones abstractas. `verify_phase7` ejecuta ocho grupos
independientes con xCoba cuando se activa la integración externa.

La fase 8 verifica el informe agregado, sus estados, reconstrucciones,
componentes, símbolos, serialización y adaptación de evidencia Wolfram. Incluye
una regresión para contracciones externas de sumas con índices mudos internos.

La fase 9 cubre paquetes, identidad por contenido, manifiestos, hashes y LaTeX.
La fase 10 añade regresiones del orquestador integral, normalización de acción,
ramas opcionales, eventos, archivos JSON y CLI.

La fase 11 cubre fingerprints, transporte IR–xAct, rechazo de evidencia ajena,
la etapa opcional Wolfram y una validación genérica en vivo con ocho residuales.

La fase 12 añade Bianchi multitémino, reducción diferencial, contrato
`adjudicates`, unanimidad de evidencia y propagación de la adjudicación al
informe y al manifiesto.

La fase 13 cubre invariantes de autoría, catálogo, serialización de campañas,
configuración uniforme, aislamiento de fallos y comandos `catalog`/`campaign`.

La fase 14 verifica equivalencia semántica fuente–catálogo, fracciones exactas,
fingerprints, límites de gramática, rechazo de construcciones ejecutables y los
comandos `compile`/`run-source`.

`test_curvature_invariants.py` cubre `RicciSq` y `RiemannSq`: equivalencia entre
alias, `ModelBuilder` y contracciones de bajo nivel; ausencia de identidades 3D
implícitas; proyección coordenada en `draft4_circular` y FLRW; reutilización del
pipeline variacional; migración conservadora de bundles antiguos y regresión de
los lagrangianos previamente soportados. La prueba viva correspondiente en
`test_wolfram_bridge.py` exige que las nueve identidades genéricas de cada modelo
se transporten y validen sin fallos ni diagnósticos de decodificación.

`test_components.py` comprueba además que `draft4_circular` use por defecto
`Phi(tau,r,varphi)`, que `p` aparezca solo tras especializar explícitamente
`phi=p*varphi`, y que las sustituciones posteriores de `f(r)` y `Phi` sobrevivan
la serialización del ansatz.
