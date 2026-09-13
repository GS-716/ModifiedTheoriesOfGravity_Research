# Fase 3: tablas comparativas

Respaldo de entrada: `backups/fase2-final-20260912-060735/`.
Incluye `main.tex`, `main.pdf` y todos los fuentes de `fase2/`, con los
mapas progresivos sin flechas. El respaldo original sigue intacto.

Se añaden ocho diapositivas. El PDF pasa de 141 a 149 páginas.

| Página | Tabla | Ubicación |
| --- | --- | --- |
| 31 | EH y cosmológico: momentos y contracción | Después de ambos controles de Lie, antes del balance |
| 55 | EH y cosmológico: ecuaciones, Bianchi y frontera | Después de ambos controles variacionales |
| 65 | EH y cosmológico: corriente y potencial | Después de ambos controles de Noether |
| 80 | Diccionario de los cuatro momentos generalizados | Después de sus definiciones y antes de variar |
| 87 | Comparación de la variación geométrica y escalar | Después de obtener la ecuación métrica generalizada |
| 116 | Momentos de Q beta: derivadas y usos posteriores | Antes de desarrollar los cuatro momentos |
| 122 | Resultados de curvatura y métrica de Q beta | Después de ambas deducciones |
| 126 | Resultados del gradiente y campo escalar de Q beta | Después de calcular Pi y F phi |

La comparación existente de Bianchi y Noether se mantiene íntegra en la
página 97. No se suprime ninguna deducción ni se modifica la secuencia de
los mapas de progreso. Los nuevos resúmenes no introducen ecuaciones numeradas.

## Diseño y preservación

Los fuentes editables están en `fase3/`. Los estilos se aplican localmente a
las tablas y conservan Madrid, sus colores y las tipografías del documento.
Las fórmulas extensas se distribuyen en líneas dentro de las celdas, sin
eliminar términos. Las tablas mantienen visibles las variables de derivación
y distinguen el momento definido respecto de g covariante del definido
respecto de g contravariante.

Se conservan exactamente el contenido anterior de `main.tex` al excluir las
nueve inclusiones nuevas, las 62 etiquetas en su orden y todos los fuentes
de `fase2/`. La comparación de las 141 diapositivas heredadas no encuentra
diferencias de imagen fuera del pie, que actualiza la numeración.

Se revisaron visualmente las ocho tablas y se corrigieron sus márgenes
internos. La compilación final no presenta avisos de desborde ni referencias
pendientes. Esta comprobación verifica conservación y presentación, no
constituye una nueva validación de la teoría de partida.

La revisión reproducible y sus imágenes están en `.review/fase3/`.
La compilación aislada usa copias de `main.tex`, `fase2/` y `fase3/` en
`.review/fase3/build/` para evitar reutilizar archivos auxiliares obsoletos.

Quedan pendientes el revelado progresivo de las deducciones y las
incorporaciones previstas para las fases posteriores.
