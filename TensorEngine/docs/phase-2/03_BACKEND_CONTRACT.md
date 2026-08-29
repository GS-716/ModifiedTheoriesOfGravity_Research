# Fase 2.3 — Contrato de backends tensoriales

`TensorBackend` define la interfaz que deberán compartir el backend estructural,
SymPy y xAct.

## Operaciones de la interfaz

- `canonicalize`;
- `simplify`;
- `expand`;
- `substitute`;
- `symmetrize` y `antisymmetrize`;
- `tensor_product`;
- `raise_index` y `lower_index`;
- `check_first_bianchi`.

## Capacidades declaradas

Cada backend publica nombre, versión y un conjunto explícito de capacidades.
El backend `structural-python` declara:

| Capacidad | Estado |
|---|---|
| Higiene de índices | soportada |
| Sustitución estructural | soportada |
| Expansión | soportada |
| Simetrías monótérmino | soportada |
| Contracción métrica directa | soportada |
| Bianchi multitémino | no soportada |
| Identidades dimensionales generales | no soportada |
| Cálculo de derivadas covariantes | no soportada al cierre de fase 2; incorporada en fase 3 |

Solicitar una capacidad ausente produce `BackendCapabilityError`; no genera un
resultado aproximado ni una igualdad asumida.

## Función del backend estructural

Este backend es una referencia portable para validar la IR, preparar solicitudes
y comprobar operaciones elementales. No sustituye al backend especializado que
se conectará posteriormente para la canonización tensorial completa.
