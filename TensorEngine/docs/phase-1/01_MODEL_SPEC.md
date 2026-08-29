# Fase 1.1 — ModelSpec

`ModelSpec` es la entrada canónica de una corrida. Es inmutable, validable y
serializable; no contiene objetos de SymPy ni expresiones de Wolfram Language.

## Campos principales

- `name`: identificador estable del modelo;
- `lagrangian`: árbol de expresión IR completamente contraído;
- `dimension`: símbolo abstracto o entero `D >= 2`;
- `normalization`: factor global escalar de la acción;
- `parameters`: constantes simbólicas y sus hipótesis;
- `functions`: funciones escalares declaradas y su aridad;
- `assumptions`: hipótesis explícitas del modelo;
- `metadata`: información descriptiva que no cambia la matemática;
- `symbols`: nombres semánticos de `g`, `Riemann`, `phi` y `u`;
- `conventions`: referencia a las convenciones de fase 0;
- `schema_version`: versión del formato serializado.

## Símbolos iniciales reservados

| Nombre IR | Significado | Varianza admitida |
|---|---|---|
| `g` | métrica inversa `g^{ab}` | `up, up` |
| `Riemann` | curvatura `R_abcd` | cuatro índices `down` |
| `phi` | campo escalar | rango cero |
| `u` | `u_a = ∇_a phi` | un índice `down` |

Además de la varianza, cada cabeza conserva su simetría semántica: `g` es
simétrica, `Riemann` tiene simetría algebraica de Riemann y `u` no tiene una
simetría adicional.

La forma completamente covariante de Riemann y la métrica inversa hacen
explícita toda contracción métrica. Esto permite separar posteriormente la
derivada respecto a `g^{ab}` de la derivada respecto a `R_abcd`.

## Ejemplo de construcción

```python
from tensor_engine import (
    FunctionSpec, ModelBuilder, ModelSpec, Number, function
)

b = ModelBuilder()

R = (
    b.metric("a", "c")
    * b.metric("b", "d")
    * b.riemann("a", "b", "c", "d")
)
X = b.metric("a", "b") * b.scalar_gradient("a") * b.scalar_gradient("b")

L = function("F", b.phi) * R - Number(1, 2) * X - function("V", b.phi)

model = ModelSpec(
    name="scalar_tensor_example",
    lagrangian=L,
    functions=(FunctionSpec("F"), FunctionSpec("V")),
)
```

Construir `ModelSpec` ejecuta validación estructural inmediatamente.

## Reglas de validación

- `L` y la normalización deben ser escalares;
- todo parámetro y función debe declararse;
- los nombres geométricos son reservados;
- las contracciones requieren el mismo índice una vez arriba y una abajo;
- un índice no puede aparecer más de dos veces en un monomio;
- las sumas deben tener los mismos índices libres término a término;
- el input inicial no acepta `CovariantDerivative`: usa `u_a`;
- las convenciones deben coincidir con `tensor-engine.phase0.v1`.
- la normalización global solo puede depender de parámetros y de `D`.
