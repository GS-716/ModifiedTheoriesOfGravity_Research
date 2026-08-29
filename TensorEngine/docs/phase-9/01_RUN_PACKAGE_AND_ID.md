# Paquete de corrida e identidad

## Objetivo

La fase 9 reúne el modelo validado, los momentos, la variación cruda, el
resultado de Euler–Lagrange, la evidencia de verificación y, si existen, los
resultados Noether–Wald y de componentes en un `RunPackage` inmutable.

## Identidad por contenido

`run_id` es el prefijo de un SHA-256 calculado sobre JSON canónico que incluye:

- `ModelSpec` y el lagrangiano normalizado;
- momentos y resultados variacionales;
- resultados opcionales de Noether y componentes;
- informe de verificación y versiones de sus fuentes externas.

La fecha, la ruta de salida y las duraciones se excluyen de esa identidad. Dos
corridas matemáticamente iguales conservan el mismo ID aunque se exporten en
otro momento o equipo. Un cambio en la entrada, los resultados o la evidencia
produce otro ID.

## Reconstrucción

`results.json` contiene los nodos originales de la IR, no cadenas LaTeX. Puede
reconstruirse con `RunPackage.from_data`, que recalcula y comprueba el `run_id`.
Esto detecta alteraciones accidentales en el contenido semántico.
