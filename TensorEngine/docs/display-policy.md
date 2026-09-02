# Presentación simplificada sin cambiar la corrida

## Uso en el notebook

La presentación conservadora está activa por defecto. No cambie la declaración
del lagrangiano ni construya ecuaciones manualmente. Puede configurar únicamente
la exportación:

~~~python
from tensor_engine import DisplayPolicy, TensorEngine

policy = DisplayPolicy(
    factor=True,
    collect=True,
    together=True,
    canonicalize_indices=True,
    aggressive=False,
)
run = TensorEngine().run(
    model, ansatz=ansatz, output_root="outputs", display_policy=policy,
)
~~~

Para conservar también la tipografía anterior use DisplayPolicy(enabled=False).
La política no se incorpora al modelo ni a la identidad de la corrida.

## Reexportar sin recalcular

~~~python
from tensor_engine import RunExporter, build_presentation

view = build_presentation(
    run.package, policy, projected_assumptions=ansatz.assumptions,
)
record = view.record("abstract.curvature_momentum")
record.canonical       # referencia a la expresión calculada, inmutable
record.presentation    # otra expresión IR, exclusivamente para presentación
record.status          # simplified, unchanged, disabled o fallback
record.operations
record.assumptions_used
record.notes

bundle = RunExporter(
    "outputs/readable", display_policy=policy,
    projected_assumptions=ansatz.assumptions,
).export(run.package)
~~~

Al usar TensorEngine.run, el exportador recibe automáticamente las hipótesis del
ansatz real. Un RunPackage antiguo conserva el nombre del ansatz, no todas sus
hipótesis: al reexportarlo, proporciónelas explícitamente si quiere usarlas.
Nunca se deducen hipótesis a partir de nombres como draft4_circular o flat_flrw.
Las hipótesis del ansatz solo afectan la presentación de componentes, no la
sección abstracta ni las expresiones abstractas de respaldo.

PresentationBuilder(model, policy).expression(expr, assumptions=()) permite
presentar una expresión aislada. La caché pertenece a una exportación y reutiliza
las expresiones repetidas; no modifica ni sustituye objetos de run.abstract,
run.projected, momentos, ecuaciones, verificaciones o backends.

## Descomposición compacta de ecuaciones y momento de curvatura

Después de las once cantidades que ya presenta cada una de las dos secciones,
el informe añade una vista de auditoría para `E_ab`, `E_phi` y `P^{abcd}`. No
reemplaza ni reordena la salida anterior. Sus bloques son genéricos:

~~~text
E_ab  = M_ab + término algebraico de curvatura
             - (1/2) g_ab L + E_ab^(nabla nabla P)
E_phi = F_phi - nabla_a J^a
P     = derivada del lagrangiano respecto de R_abcd
~~~

El término algebraico y el término de derivadas son exactamente los objetos que
usa `metric_euler_expression`; `-nabla_a J^a` se recupera como la diferencia
canónica ya calculada `E_phi-F_phi`. Por tanto, esta vista no deriva nuevamente
las ecuaciones. Cada bloque conserva una forma compacta de presentación, la IR
expandida para auditoría, sus fuentes y el resultado de reconstrucción.

Las formas expandidas breves se muestran también en LaTeX/PDF. Si una expansión
supera el umbral seguro de página, el reporte muestra solo la forma compacta y
una referencia a `presentation.json`, donde la expansión permanece completa.
No se introducen símbolos auxiliares del tipo `A_1 + A_2 + ...` en la salida
final.

`ReportPresentation.compact_decompositions` permite consultar los tres grupos y
sus bloques. En `presentation.json` aparecen bajo `compact_decompositions`, con
estados separados para reconstrucción abstracta y por componentes. Una corrida
activa pasa al exportador el backend del ansatz y proyecta los bloques. Al
reexportar solamente un `RunPackage`, la geometría completa del ansatz no está
persistida: los bloques intermedios que no eran resultados de primera clase se
conservan simbólicos con ese motivo explícito. Los objetivos y los bloques ya
almacenados (`M_ab`, `F_phi` y `P`) se siguen reutilizando.

La comprobación de reconstrucción usa suma IR canónica y, para componentes,
álgebra escalar exacta. No conmuta derivadas, no usa ecuaciones de campo, no
introduce etiquetas físicas inferidas y no participa en resultados, hashes,
manifiestos ni evidencia xAct.

## Archivos y trazabilidad

- results.json y verification.json conservan exactamente sus datos y formato.
- presentation.json es un archivo adicional con la IR de presentación, LaTeX,
  estado, operaciones, hipótesis usadas, notas y SHA-256 de la expresión
  canónica de origen. Su propósito es presentation_only.
- El mismo archivo contiene las descomposiciones compactas adicionales, sus
  bloques, formas expandidas, componentes y estados de reconstrucción.
- El registro cubre las once cantidades abstractas, la contribución métrica
  de nabla_nabla_P y **todas** las componentes dispersas obtenidas, incluso las
  que no se imprimen por el límite de doce componentes por tensor. Los ceros
  implícitos siguen siendo ceros; las cantidades no proyectables mantienen su
  respaldo abstracto.
- Los comentarios del .tex incluyen el registro de cada expresión impresa.
  Las operaciones tipográficas se distinguen de las operaciones algebraicas.
- El PDF mantiene exactamente dos secciones principales. No vuelve a evaluar
  ninguna ecuación de campo, derivada covariante o geometría coordenada.

El esquema del manifiesto, run_id, hashes matemáticos, proyecciones y enlaces de
evidencia xAct no cambian. El inventario de archivos añade presentation.json.
Los hashes de report.tex/report.pdf necesariamente cambian si cambia su contenido;
se actualizan mediante el mecanismo de integridad existente. Conservar el hash
antiguo de un PDF modificado invalidaría el manifiesto.

El SHA-256 canónico del registro usa JSON ordenado compacto, el mismo criterio
empleado para expresiones por el exportador. La vista de presentación no se
deserializa como resultado científico ni se envía a Wolfram/xAct.

## Operaciones seguras

1. Renombramiento higiénico de índices mudos. Las simetrías y contracciones de
   métricas/deltas se resuelven antes, en la IR canónica: DisplayPolicy no hace
   contracciones matemáticas independientes ni llama al backend estructural.
2. Agrupación de términos semejantes, aritmética racional exacta y normalización
   de signos. Las contracciones tensoriales completas son bloques opacos al
   álgebra escalar: no se factorizan componentes de un tensor como si fueran
   escalares independientes.
3. Recolección por parámetros declarados, extracción de factores comunes y
   combinación de fracciones justificadas. Se selecciona una alternativa solo
   cuando reduce el tamaño estructural; una fracción común mayor puede descartarse.
4. aggressive=True añade búsqueda de factorización polinómica; **no** permite
   cancelar factores sin condiciones, conmutar derivadas ni alterar ramas.
5. Tipografía: elimina el coeficiente redundante -1, imprime potencias negativas
   como fracciones sin cancelarlas, usa ell/beta0 como símbolos matemáticos,
   agrupa índices contiguos de igual varianza preservando los slots y muestra
   derivadas univariadas de orden 1/2 con primas.

La política no usa simplify indiscriminadamente ni evalúa funciones arbitrarias.
Las funciones, derivadas formales y potencias no justificadas se tratan como
bloques. Las operaciones escalares candidatas se comprueban algebraicamente en
esa representación protegida; también se verifica la firma de índices libres.

## Hipótesis y límites

Se reconocen condiciones explícitas de la forma ell!=0, r>0, r<0, a(t)>0 o
f(r)!=0. También se reconocen nonzero, positive y negative en ParameterSpec.
Las condiciones real o r>=0 **no** autorizan una cancelación. Las condiciones
no reconocidas se conservan en los datos originales, pero no se interpretan.
No se usa eval ni un parser de código para las hipótesis.

Sin ell!=0, tanto ell/ell como 1/ell - 1/ell conservan sus denominadores: la vista
no elimina su singularidad. También se protege esa condición durante la
canonización tensorial. Las potencias fraccionarias no reciben identidades de
ramas; las derivadas covariantes no se conmutan ni se vuelven a calcular.
Las hipótesis usadas para habilitar álgebra racional se registran por expresión.

max_nodes=4000 limita la búsqueda antes de iniciarla. Si una operación no está
soportada o falla, se conserva la expresión original con estado fallback y el
motivo. No es un límite de tiempo duro; expresiones dentro del presupuesto aún
pueden ser costosas. El modo polinómico ampliado es opcional por esa razón.

La simplificación no crea componentes que el backend no calculó. La limitación
de delta en la etapa de ecuaciones se resolvió en el álgebra canónica y en el
backend de componentes (véase [contracciones delta](delta-contractions.md)).
Permanece el límite de 2048 componentes potenciales. La identidad diferencial de Noether
puede seguir indeterminada. Que una expresión sea más corta no implica una
nueva validación independiente con xAct.

Las expresiones ya compactas, los ceros, R_abcd como tensor de entrada, las
potencias protegidas y muchos componentes ya factorizados pueden no reducirse
algebraicamente; aun así reciben la tipografía y el registro de presentación.

## Verificación histórica de la entrega de presentación

Los resultados siguientes corresponden a la entrega anterior a la extensión
de contracciones delta; no describen las limitaciones actuales de proyección.

- Suite completa con TENSOR_ENGINE_RUN_WOLFRAM_TESTS=1: 290 pruebas aprobadas
  en 328.61 s. Incluye Caso-2 con draft4_circular y FLRW, y un ansatz de usuario.
- Después de corregir una fracción larga detectada visualmente se ejecutó la
  regresión de presentación/exportación: 36 pruebas aprobadas, incluida una
  nueva prueba de numeradores largos que no se pueden partir dentro de frac.
- results.json y verification.json de ambas reexportaciones son idénticos byte
  por byte a los de origen. Los manifiestos solo difieren en fecha e inventario
  de archivos; sus campos matemáticos y vínculos xAct son iguales.
- La ida y vuelta RunPackage JSON y las verificaciones SHA-256 de ambos bundles
  están comprobadas. Cada reporte mantiene dos secciones y veintidós cantidades
  principales, once abstractas y once proyectadas/con respaldo abstracto.
- Los dos PDF de doce páginas se renderizaron y revisaron visualmente completos.
  No quedaron expresiones recortadas ni superpuestas. Los reportes anteriores
  se conservaron; las copias nuevas están en outputs/presentation.
- La longitud LaTeX de la vista abstracta, incluyendo el término métrico de
  nabla_nabla_P, pasó de 18434 a 13312 caracteres (27.8 % menos). Es una medida
  de longitud de expresiones, no una reducción del número de páginas.

El registro de draft4 contiene 125 expresiones (13 con reducción algebraica),
y el de FLRW 143 (18 con reducción). Las demás conservan su forma algebraica y
reciben formato. Ninguna tuvo estado fallback de presentación en estas corridas.
Las limitaciones de proyección y las dos comprobaciones indeterminadas de
Noether permanecen explícitas; no se reclasificaron como éxitos.

Archivos de esta extensión: src/tensor_engine/presentation.py,
src/tensor_engine/exporting.py, src/tensor_engine/engine.py,
src/tensor_engine/__init__.py, tests/test_presentation.py,
tests/test_exporting.py, tests/test_source_integration.py, README.md,
docs/display-policy.md y docs/phase-9/03_JSON_AND_LATEX.md.
