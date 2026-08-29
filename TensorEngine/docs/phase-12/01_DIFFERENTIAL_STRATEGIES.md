# Estrategias diferenciales genéricas

La fase 12 conserva un único transporte IR–xAct y selecciona la reducción por
medio del campo enumerado `strategy` de cada comprobación:

| Estrategia | Uso | Reducción |
|---|---|---|
| `algebraic` | simetrías y sumas exactas | `ContractMetric` + `ToCanonical` |
| `riemann_bianchi` | identidad cíclica de \(P^{abcd}\) | proyección de Young de Riemann |
| `differential` | Noether y divergencias | ordenamiento de derivadas, canonización y Bianchi contraída |

La estrategia diferencial prueba, para el modelo escalar–tensor de referencia,

\[
J_\xi^a-2E^a{}_b\xi^b-\nabla_bQ_\xi^{ab}=0,
\qquad
2\nabla^aE_{ab}+E_\phi\nabla_b\phi=0.
\]

El delta de Kronecker del IR se transporta como métrica mixta de la conexión de
Levi-Civita. Así xAct puede contraerlo y demostrar su compatibilidad métrica; no
se declara como un campo tensorial independiente.
