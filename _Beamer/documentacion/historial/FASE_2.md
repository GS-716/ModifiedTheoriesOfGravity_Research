# Segunda fase: diagramas de flujo

Nota: este registro describe la primera implementación. La revisión vigente,
con mapas completos y seguimiento progresivo, está en `FASE_2_REVISION.md`.

Respaldo de entrada: `backups/fase1-20260912-000458/main.tex` y `main.pdf`.
El respaldo original previo a la fase 1 permanece en su carpeta original.

Se añaden cuatro diapositivas editables en TikZ, conservando Madrid y su paleta:

| Página del PDF | Diagrama | Ubicación y finalidad |
| --- | --- | --- |
| 7 | Identidad principal | Entrada de la sección de Lie. Dos cálculos convergen en la comparación y en la identidad entre los momentos. |
| 28 | Variación de la acción | Entrada de la sección variacional. Muestra cómo Palatini y las integraciones por partes separan volumen y frontera. |
| 55 | Criterio de segundo orden | Entrada de la sección correspondiente. Relaciona el posible orden superior con la condición de divergencia nula. |
| 106 | Comprobación sobre el ansatz | Después del cálculo explícito y antes del balance. Reúne el sector relevante, el ansatz y la componente calculada. |

Las diapositivas previas mantienen su contenido y orden. Los diagramas anticipan
o recapitulan resultados ya presentes, sin sustituir las deducciones.
El documento pasa de 104 a 108 páginas.

Los archivos de los diagramas y sus estilos están en `fase2/`.
`main.tex` los incorpora mediante cinco instrucciones `input`.
Al retirar estas inclusiones y el comentario de fase 2, el contenido de
`main.tex` coincide con el respaldo de fase 1, salvo espacios en blanco.

Verificación: compilación final sin avisos de desborde ni referencias pendientes,
revisión visual de los cuatro diagramas y ajustes de las bifurcaciones y espacios.
La comparación de imágenes de las 104 diapositivas heredadas muestra identidad
fuera del pie de página en las 103 posteriores a la portada. La portada actualiza
su fecha automática. La numeración del pie refleja las cuatro incorporaciones.

Las tablas comparativas y los gráficos de las siguientes fases quedan pendientes.
