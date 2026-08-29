# Fase 0.3 — Contrato de salida de una corrida

Cada corrida debe producir un registro estructurado y reproducible. Los nombres
que aparecen aquí son nombres semánticos; su representación concreta se fijará
en la fase de diseño de `ModelSpec`.

## 1. Manifiesto obligatorio

- identificador único de corrida;
- versión del esquema de resultados;
- fecha y duración;
- backend y versiones empleadas;
- dimensión abstracta o concreta;
- firma y convención de Riemann;
- lagrangiano original y lagrangiano normalizado;
- campos, parámetros e hipótesis declaradas;
- estado final: `success`, `failed` o `partial`.

## 2. Objetos variacionales obligatorios

La corrida abstracta debe conservar:

- `lagrangian`: `L`;
- `metric_momentum`: `M_ab`;
- `curvature_momentum`: `P^{abcd}`;
- `scalar_gradient_momentum`: `J^a`;
- `scalar_derivative`: `F_φ`;
- `delta_lagrangian`: regla de la cadena antes de integrar por partes;
- `metric_euler`: `E_ab`;
- `scalar_euler`: `E_φ`;
- `boundary_potential_metric`: contribución métrica a `Θ^a`;
- `boundary_potential_scalar`: contribución escalar a `Θ^a`;
- `boundary_potential_total`: `Θ^a`;
- `full_variation`: variación completa de la acción.

Cuando existan formas algebraicamente equivalentes, se almacenarán por separado
como `raw`, `canonical` y `model_reduced`; ninguna sobrescribirá a las demás.

## 3. Verificaciones obligatorias

Toda corrida debe informar, como mínimo:

- simetría de `M_ab`;
- simetrías algebraicas de `P^{abcd}`;
- simetría de `E_ab`;
- consistencia de la integración por partes escalar;
- equivalencia entre las formas raw y canonical;
- identidad de Bianchi–Noether off-shell;
- ausencia de índices libres inesperados;
- ausencia de símbolos no declarados;
- estado de cada verificación: `passed`, `failed` o `undetermined`.

Una verificación `undetermined` no cuenta como aprobada y debe conservar la
expresión residual que el backend no pudo decidir.

## 4. Resultados extendidos

Las fases posteriores podrán añadir sin modificar el núcleo del contrato:

- derivadas de Lie;
- corriente de Noether;
- potencial de carga;
- componentes en una geometría concreta;
- evaluación sobre un ansatz;
- límites y ramas de soluciones;
- resultados numéricos;
- LaTeX, PDF y figuras.

## 5. Trazabilidad por etapas

Cada objeto calculado debe registrar:

- clave estable;
- etapa productora;
- objetos de entrada;
- operación aplicada;
- backend responsable;
- hipótesis usadas;
- expresión anterior y expresión resultante;
- tiempo de cálculo;
- verificaciones asociadas.

Los notebooks solo presentan estas etapas. No serán la única ubicación donde
exista una expresión o una demostración.

## 6. Política de errores

- Un error matemático o de backend termina la etapa con diagnóstico explícito.
- Un resultado parcial no se presenta como derivación completa.
- No se reemplaza automáticamente una expresión problemática por cero.
- Las excepciones específicas de un modelo deben declararse en su especificación.
- Los artefactos de presentación no son la fuente canónica de los resultados.

