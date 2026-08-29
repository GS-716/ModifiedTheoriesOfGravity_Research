# Contrato de ansatz geométrico

## Separación entre teoría y geometría

`ModelSpec` describe el lagrangiano covariante y `GeometryAnsatz` describe una
realización coordenada. Son entradas separadas deliberadamente: un mismo modelo
puede evaluarse sobre varias geometrías y un mismo ansatz puede reutilizarse con
varios modelos. Las fases 0–6 permanecen independientes de coordenadas.

Un ansatz contiene:

- una `CoordinateChart` ordenada;
- la matriz covariante simétrica `g_ab` en expresiones de la IR;
- un campo escalar opcional;
- hipótesis declarativas y versión de esquema.

El contrato exige dimensión entera, coincidencia entre la dimensión del modelo
y la carta, matriz cuadrada y no degenerada, y componentes escalares. Tanto la
carta como el ansatz admiten ida y vuelta JSON sin introducir código evaluable.

## Referencia de aceptación

La función `spatially_flat_flrw_ansatz()` define

\[
ds^2=-dt^2+a(t)^2(dx^2+dy^2+dz^2),\qquad \phi=\phi(t),
\]

con firma `mostly_plus` y dimensión cuatro.
