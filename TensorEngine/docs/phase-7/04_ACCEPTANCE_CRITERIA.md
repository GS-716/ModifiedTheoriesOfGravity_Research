# Criterios de aceptación de fase 7

La fase queda cerrada cuando:

- el ansatz y los resultados por componentes son serializables;
- se rechazan dimensiones simbólicas o incompatibles antes del cálculo;
- la geometría FLRW reproduce Christoffel, Ricci, `R`, Einstein y `Box(phi)`;
- la proyección abstracta de `R`, `G_ab` y `Box(phi)` coincide con el cálculo
  coordenado directo;
- la selección de ecuaciones independientes conserva también componentes cero;
- la suite Python completa no introduce regresiones;
- Wolfram Engine/xCoba aprueba los ocho grupos independientes sin residuales.

Quedan fuera de esta fase la resolución automática de ODE/PDE, condiciones de
frontera concretas, reducción por simetrías mediante órbitas y optimizaciones
para tensores densos de dimensión alta.
