# Fase 3.3 — Integración con el backend

`TensorBackend` incorpora ahora métodos para:

- `covariant_derivative`;
- `gradient`;
- `hessian`;
- `divergence`;
- `laplacian`;
- `lie_derivative`;
- verificación del conmutador;
- verificación de Bianchi diferencial.

`DifferentialContext` contiene los nombres de la métrica, curvatura, campo
escalar, gradiente escalar, vector de Lie, dimensión y escalares constantes.
Puede construirse directamente desde `ModelSpec`.

El backend `structural-python` declara soporte para:

- cálculo covariante formal;
- regla de la cadena funcional;
- construcción del conmutador y su acción de curvatura;
- derivada de Lie abstracta.

No declara reducción general de Bianchi diferencial. Una solicitud explícita de
esa capacidad genera `BackendCapabilityError`; el método de verificación, en
cambio, devuelve un residual auditable con estado `undetermined`.

