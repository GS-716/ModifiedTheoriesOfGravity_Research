# FieldEquationsSolver

Este paquete contiene la reducción, clasificación y resolución formal de las
ecuaciones de campo producidas previamente por TensorEngine. No vuelve a derivar
el lagrangiano ni modifica la corrida tensorial de origen.

## Organización

| Ruta | Responsabilidad |
|---|---|
| `src/field_equations_solver/solving.py` | Construcción de $E^a{}_b$, reducción, clasificación y verificación de soluciones |
| `src/field_equations_solver/bridge.py` | Transporte local dedicado hacia Wolfram Engine |
| `src/field_equations_solver/reporting.py` | Bundle JSON, manifiesto y reporte LaTeX/PDF del solucionador |
| `wolfram/FieldEquationSolver.wl` | Adaptador autónomo para `Solve`, `Reduce`, `Eliminate` y `DSolve` |
| `tests/` | Pruebas unitarias, integración y validaciones optativas con Wolfram/xAct |
| `examples/` | Ejecuciones reproducibles fuera del notebook |

La dependencia va únicamente desde FieldEquationsSolver hacia TensorEngine. La
validación tensorial xAct permanece en TensorEngine y el solucionador registra
esa evidencia sin reinterpretarla.

`TensorEngine().run(...)` conserva su comportamiento. Una llamada posterior
consume las componentes calculadas y la métrica almacenada:

```python
from tensor_engine import AnsatzSpecialization, Scalar
from field_equations_solver import (
    FieldEquationWolframBridge,
    SolverSearchPolicy,
    solveFieldEquations,
)

solucion = solveFieldEquations(
    run,
    specialization=AnsatzSpecialization(
        scalar_field=Scalar("q") * Scalar("varphi"),
    ),
    search_policy=SolverSearchPolicy(),
    output_root="outputs/field_equations",
)
```

Este ejemplo impone exactamente `phi = q*varphi`, sin constante aditiva.
`f(r)` sigue siendo una incógnita. Si `run` ya contiene una especialización de
`f`, por defecto se utiliza su **proyección genérica**. Para consumir sus
componentes especializadas debe solicitarse `use_specialized=True`.

Omitir `specialization` conserva el campo original. Para un campo radial se usa
`AnsatzSpecialization(scalar_field=Function("Phi", (Scalar("r"),)))`.
La misma validación existente impide reintroducir el tiempo en Draft 4.
Las sustituciones se aplican a las componentes existentes, incluidas sus
derivadas; no se repiten la derivación abstracta ni la proyección tensorial.

Si faltan componentes genéricas y el usuario solicita un perfil evaluable, solo
las componentes ausentes se proyectan por primera vez desde las ecuaciones
abstractas guardadas. Esto se registra en `classification.projection_source`.
Sin especialización se respeta el límite del backend y se conservan las
ecuaciones que sí están disponibles. `use_specialized` y una especialización
nueva son opciones excluyentes; un perfil ya sustituido no permite reconstruir
automáticamente su campo genérico.

## Opciones y resultados

- `solve=False`: construye y clasifica el sistema, sin buscar candidatos ni
  ejecutar Wolfram.
- `search_policy=SolverSearchPolicy(...)`: controla la búsqueda acotada de
  constantes, factores, ramas singulares, polinomios, potencias y escenarios de
  parámetros. El valor predeterminado es exhaustivo respecto de esas clases,
  pero nunca afirma exhaustividad matemática.
- `wolfram_bridge=FieldEquationWolframBridge(timeout_seconds=180)`: transporte y límite
  de tiempo configurables. Sin instalación disponible, las ramas locales se
  conservan y Wolfram queda marcado como no disponible.
- `eliminate=(Scalar("alpha"),)`: solicita eliminar explícitamente incógnitas;
  la relación eliminada es una consecuencia, no reemplaza la verificación original.
- `compile_pdf=False`: exporta JSON y LaTeX sin compilar PDF.
- `display_policy=DisplayPolicy(...)`: reutiliza la presentación existente.
- `solve_field_equations` es el alias Python de `solveFieldEquations`.

`solucion.original_equations` conserva todas las componentes covariantes
`E_ab` y `E_phi`, incluidas las nulas. `mixed_components` contiene
`g^{ac} E_cb` con el primer índice elevado mediante la **inversa completa** de
la métrica, también cuando hay componentes no diagonales.

`solucion.equations` conserva las diferencias diagonales, una diagonal absoluta,
todas las componentes fuera de la diagonal y la ecuación escalar. En Draft 4 las
diferencias son `(tau,tau)-(r,r)`, `(tau,tau)-(varphi,varphi)` y
`(r,r)-(varphi,varphi)`. Las dos últimas no reemplazan la ecuación absoluta.
Para otra dimensión se generan todos los pares diagonales.

## Alcance matemático de la reducción

Se detectan ceros y dependencias lineales **exactas con coeficientes racionales**.
Cada dependencia conserva su certificado. `linearly_independent_over_Q` solo
afirma independencia en ese sentido: no certifica independencia diferencial,
identidades de Noether ni ausencia de relaciones no lineales. Una relación no
demostrada se conserva. No se divide por un parámetro para descartar ecuaciones.

La clasificación estructural distingue `algebraic`, `ODE`, `PDE`, `DAE` y `mixed`,
e informa de incógnitas, variables, órdenes, restricciones y funciones libres no
determinadas. Una PDE acoplada a una EDO se etiqueta `mixed` y `contains_pde=True`.
Los factores nulos son candidatos a ramas que requieren verificar el sistema
completo; no son automáticamente soluciones independientes.

La política predeterminada separa los casos `alpha=0`/`alpha!=0`,
`q=0`/`q!=0`, `beta0=0`/`beta0!=0` y exige `ell!=0` cuando esos símbolos están
presentes. También prueba `f(r)=C_f` con `C_f!=0`, registra `f(r)=0` como rama
degenerada y examina ansätze polinomiales de grados 1 y 2 y potencias enteras
acotadas. Cada intento y su motivo quedan en `search_summary`; una clase que no
puede reducirse con seguridad permanece `pending`.

El adaptador de coordenadas utiliza el transporte JSON existente hacia Wolfram:
`Solve` y `Reduce` trabajan las relaciones algebraicas (tratando los valores de
funciones y derivadas como variables auxiliares cuando corresponde); `Eliminate`
solo elimina las incógnitas solicitadas; `DSolve` intenta integrar sin condiciones
iniciales o de frontera. Las relaciones algebraicas entre derivadas no certifican
integrabilidad. No se llama a ningún método numérico.

Las salidas no evaluadas, los límites de tiempo y las funciones que no se pueden
traducir de vuelta se guardan íntegramente como evidencia textual de Wolfram y
diagnósticos. La referencia para las operaciones formales es la documentación de
[DSolve](https://reference.wolfram.com/language/ref/DSolve.html),
[Reduce](https://reference.wolfram.com/language/ref/Reduce.html),
[Solve](https://reference.wolfram.com/language/ref/Solve.html) y
[Eliminate](https://reference.wolfram.com/language/ref/Eliminate.html).

## Verificación y persistencia

Cada candidato se sustituye en **todas las ecuaciones covariantes originales**,
incluidas las fuera de la diagonal y la escalar, y también en todas las
componentes mixtas almacenadas $E^a{}_b$. Ambos conjuntos de residuales se
conservan por componente.
Una solución solo recibe `verified_on_domain` si todas las sustituciones son
nulas y no se detecta un dominio singular. Los factores que no pueden anularse
siguen siendo restricciones explícitas del dominio; no se extiende una solución
a un horizonte o a un denominador nulo mediante cancelación.

Los supuestos escalares sencillos (igualdades, desigualdades y signos de los
parámetros) se comprueban y se transmiten a Wolfram. Un supuesto no interpretable
queda registrado e impide aprobar automáticamente la familia; no se ignora.

Puede comprobarse una familia propuesta sin resolver de nuevo:

```python
from tensor_engine import Function, Scalar
r, ell, c = Scalar("r"), Scalar("ell"), Scalar("constant")
chequeo = solucion.verify({Function("f", (r,)): r**2/ell**2 + c})
print(chequeo.status, chequeo.residuals)
```

`verified_family_found` se reserva para una cobertura realmente certificada.
En búsquedas simbólicas ordinarias se usa `verified_with_pending_branches` si hay
al menos una familia verificada, `partially_solved` si solo existen candidatos
parciales o indeterminados y `no_verified_candidate` si ninguno fue aprobado.
`search_summary.completeness_proven` permanece falso mientras no exista una
demostración de completitud. No se imponen condiciones iniciales ni de frontera.

La validación xAct de la teoría se conserva en `source_results`. Verificar una
familia de funciones coordenadas no transforma una identidad xAct indeterminada
en aprobada. Las comprobaciones de sustitución Wolfram y Python están separadas
de esa evidencia tensorial.

La exportación genera un **bundle de resolución separado** con `results.json`,
`presentation.json`, `manifest.json`, `report.tex` y PDF cuando está disponible.
Su JSON contiene una copia del paquete tensorial sin cambios (`source_results`),
su fingerprint, las ecuaciones completas, las relaciones de dependencia, las
familias y sus residuales. Se reconstruye mediante
`FieldEquationSolution.from_data(data)`. Los archivos y fingerprints de la
corrida tensorial anterior no se modifican.

El reporte de resolución muestra solo combinaciones, sus expresiones reducidas,
ecuaciones absolutas y escalares necesarias, familias y estados de verificación.
El reporte tensorial original conserva su contenido habitual.
