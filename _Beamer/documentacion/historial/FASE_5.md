# Fase 5: cierre estático y lectura del resultado

Por decisión del usuario, se retiró el revelado de la fase 4: las 44 pausas
ya no están en el documento activo. Los mapas intermedios sin flechas y las
ocho tablas de las fases 2 y 3 se conservan íntegros.

Respaldo anterior a la retirada: `backups/fase4-retirada-20260912-110759/`.
Contiene los fuentes y los PDF de exposición y lectura de la fase 4.
El registro `FASE_4.md` queda identificado como histórico.

## Incorporaciones

El PDF actual contiene 151 diapositivas estáticas, una por página.

- Página 147: gráfico de la componente radial ya calculada, después de
  su deducción y antes del balance de la discusión. Se representa su valor
  absoluto normalizado respecto de un radio de referencia positivo.
  La curva es `(r/r0)^(-3)`, sin fijar valores de parámetros físicos.
  Los puntos marcados son `1/8` a `2 r0` y `1/64` a `4 r0`.
  Se distingue el decaimiento asintótico de la anulación a radio finito,
  y una componente del tensor de una norma tensorial.
- Página 150: síntesis de la identidad de Lie, el criterio LL, la extensión
  escalar y el resultado para el acoplamiento, antes de las referencias.

Los fuentes editables están en `fase5/`. El gráfico utiliza TikZ ya presente
en el proyecto. La normalización es únicamente una reexpresión del resultado
existente para su visualización, no una hipótesis física adicional.

## Preservación y revisión

Al excluir las dos inclusiones nuevas, `main.tex` coincide con el respaldo
de la fase 3 salvo espacios en blanco. Se mantienen las 62 etiquetas y
todos los fuentes de las fases 2 y 3. Las 149 diapositivas heredadas
coinciden visualmente con la fase 3 fuera del pie, que actualiza la numeración.
No queda ninguna pausa ni instrucción de revelado en los fuentes activos.

Se comprobaron la normalización del gráfico, sus valores marcados, la lectura
de ambas diapositivas nuevas y la compilación sin avisos de desborde ni
referencias pendientes. Los controles e imágenes están en `.review/fase5/`.

`main.pdf` es la versión principal. Para evitar que el enlace anterior de
lectura quede desactualizado, `main-lectura.pdf` contiene una copia idéntica
del PDF estático actual; ya no son versiones distintas.
