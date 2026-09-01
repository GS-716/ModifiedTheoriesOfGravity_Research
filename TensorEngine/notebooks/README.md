# Notebooks

Esta carpeta contendrá interfaces de investigación y ejemplos reproducibles.
Los notebooks llamarán al motor; no contendrán su lógica esencial.

Ejemplo mínimo ejecutable:

- `01_quickstart_tensor_engine.ipynb`: cuatro celdas para probar el Caso-2 EQT,
  `R + 2/ell**2 + ell**2*beta0*(3*RicciUU - X*R)`, declarado como texto.
  El motor expande `RicciUU = R_ab u^a u^b`. El ansatz predeterminado es el axial 3D
  del `Draft_4`, con `phi=p*varphi`. La opción FLRW es una extensión a 4D, no
  el caso original. Se valida con Wolfram/xAct y se exporta el bundle reproducible.
  La prueba omite la constante global `1/(16*pi*G)` y no incluye `sqrt(|g|)`
  dentro del lagrangiano escalar. La última celda muestra los estados de las
  once proyecciones y enlaces al PDF, LaTeX, JSON y manifiesto.

Solo cambia la expresión, sus declaraciones y el ansatz en la segunda celda.
`VALIDAR_XACT` en la tercera configura la validación externa. Recarga el notebook
y reinicia el kernel tras actualizar el paquete. Los estados parciales y motivos
se muestran sin ocultarlos. La [guía del frontend](../docs/frontend-invariants.md)
describe los alias, las contracciones genéricas, los registros personalizados y
las limitaciones actuales.

Desde la fase 10, la interfaz recomendada es:

```python
from tensor_engine import RunEvent, TensorEngine

events = []
run = TensorEngine(event_handler=events.append).run(
    model,
    ansatz=ansatz,          # opcional
    output_root="outputs", # opcional
)
```

El notebook puede presentar `run.package`, `run.stages` y los eventos, pero no
debe reimplementar derivaciones ni reglas de simplificación.

Desde la fase 13 también puede construir invariantes y campañas sin escribir la
IR tensorial a mano:

```python
from tensor_engine import ModelBuilder, catalog_model

b = ModelBuilder()
R = b.ricci_scalar()
X = b.kinetic_scalar()
model = catalog_model("k_essence", name="mi_modelo")
```

La fase 14 permite que un notebook reciba una fórmula textual validada:

```python
from tensor_engine import LagrangianSourceSpec

source = LagrangianSourceSpec("mi_teoria", "F(phi)*R - Z(phi)*X/2 - V(phi)", ...)
model = source.compile()
```

El texto nunca se evalúa como código Python; solo se acepta la gramática
algebraica documentada.
