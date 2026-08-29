# Notebooks

Esta carpeta contendrá interfaces de investigación y ejemplos reproducibles.
Los notebooks llamarán al motor; no contendrán su lógica esencial.

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
