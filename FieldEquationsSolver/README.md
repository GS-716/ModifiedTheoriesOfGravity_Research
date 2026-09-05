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
from field_equations_solver import FieldEquationWolframBridge, solveFieldEquations

solucion = solveFieldEquations(
    run,
    specialization=AnsatzSpecialization(
        scalar_field=Scalar("q") * Scalar("varphi"),
    ),
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

- `solve=False`: construye y clasifica el sistema, sin ejecutar Wolfram.
- `wolfram_bridge=FieldEquationWolframBridge(timeout_seconds=180)`: transporte y límite
  de tiempo configurables. Sin instalación disponible, el resultado es simbólico.
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

Cada candidato se sustituye en **todas las ecuaciones originales**, incluidas
las fuera de la diagonal y la escalar. Se conservan los residuales por componente.
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

`formal_family` indica que existe una familia verificada; no certifica que se
hayan encontrado todas las ramas singulares. `partial`, `symbolic`,
`underdetermined` y `unavailable` conservan el trabajo disponible sin afirmar una
solución completa. No se imponen condiciones iniciales ni de frontera.

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
