# Fase 1.2 — Representación intermedia tensorial

La representación intermedia (IR) es el idioma común entre Python, SymPy y
xAct. Conserva semántica tensorial y evita comunicar los modelos mediante texto
LaTeX o código evaluable.

## Nodos disponibles

- `Number`: racional exacto;
- `Scalar`: símbolo escalar;
- `Tensor`: cabeza tensorial con índices ordenados;
- `Add`: suma tensorial;
- `Mul`: producto y contracción de Einstein;
- `Power`: potencia de una expresión escalar;
- `Function`: función simbólica con argumentos escalares;
- `CovariantDerivative`: nodo reservado para resultados y fases posteriores.

La IR se amplió de manera compatible en fases posteriores con:

- `FunctionDerivative`: derivadas parciales formales de funciones;
- `Variation`: variación formal de un objeto;
- `VolumeElement`: densidad `sqrt(-g)`, distinta de un escalar ordinario.

Cada índice registra tres datos:

```text
nombre + varianza (up/down) + espacio de índices
```

## Invariantes de integridad

1. Los racionales se normalizan exactamente.
2. Los nodos son inmutables.
3. Toda expresión se serializa a diccionarios y listas JSON-compatibles.
4. La deserialización vuelve a validar los índices.
5. La IR no simplifica usando ecuaciones de campo o identidades no declaradas.
6. La forma serializada no ejecuta código en ningún backend.
7. Una subexpresión escalar delimita el alcance de sus índices mudos; antes de
   aplanar productos, la fase de canonización deberá renombrarlos con seguridad.

## Límite de esta fase

La IR valida estructura e índices, pero todavía no:

- canoniza índices mudos;
- aplica simetrías de Riemann;
- deriva expresiones;
- integra por partes;
- convierte expresiones a SymPy o xAct.

Esas capacidades consumirán esta representación en fases posteriores.
