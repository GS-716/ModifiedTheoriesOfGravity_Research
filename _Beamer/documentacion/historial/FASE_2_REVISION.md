# Fase 2 revisada: cobertura y seguimiento progresivo

Respaldo previo a esta revisión: `backups/fase2-20260912-053159/`,
con `main.tex`, `main.pdf` y los fuentes de `fase2/`.
El respaldo original permanece intacto.

Se sustituyen los cuatro mapas anteriores por seis mapas de sección, con
37 apariciones en total: al inicio, después de cada bloque desarrollado y
al cierre. El PDF pasa de 108 a 141 páginas. No se avanza a la fase 3.

## Cobertura y puntos de retorno

| Sección | Bloques | Páginas del mapa, desde cero hasta todos completados |
| --- | --- | --- |
| Lie | Herramientas; argumentos y simetrías; contracción y segundo cálculo; identidad; EH; cosmológico y balance | 7, 11, 19, 22, 25, 29, 32 |
| Acción | Palatini; integraciones y frontera; Bianchi y LL; controles EH y cosmológico; Noether; controles y balances | 34, 41, 46, 50, 54, 60, 65 |
| Segundo orden | Diagnóstico; divergencia nula y materia; resultado LL | 67, 69, 71, 73 |
| Extensión escalar | Momentos; identidad y variación; Bianchi; Noether; comparación geométrica; desplazamiento y ecuaciones | 75, 79, 83, 86, 91, 93, 98 |
| EQT | Cinético; reconstrucción del sector base; momentos de curvatura y métrica; momentos escalares; ecuaciones y frontera del acoplamiento; Bianchi y teoría completa | 100, 106, 109, 116, 119, 125, 128 |
| Discusión | Sector relevante; divergencia general; ansatz y componente; resultado y límites | 130, 132, 135, 137, 140 |

Cada cuadro incluye un encabezado y un recordatorio técnico. Los estados son:
verde con marca de verificación para lo completado, azul para el siguiente
bloque y blanco para lo pendiente. La numeración sigue la exposición, no
pretende convertir todos los pasos en implicaciones lógicas. La posición y
el contenido de los cuadros permanecen constantes entre retornos.

Los mapas activos se definen en `fase2/mapas-progreso.tex`; `main.tex` marca
los puntos de inserción. Los cuatro archivos de mapas anteriores se conservan,
pero ya no se incluyen en el documento.

## Preservación y control de calidad

- Comparación de fuentes: al excluir las inclusiones de mapas nuevas y antiguas,
  el documento coincide con el respaldo de entrada, salvo espacios en blanco.
- Se conservan las 62 etiquetas y su orden.
- Las 104 diapositivas heredadas mantienen contenido y orden. Su comparación
  visual con el PDF previo es idéntica fuera del pie, cuya numeración cambia.
- Comprobación automática de la secuencia completa de estados de cada mapa.
- Revisión visual de las 37 apariciones y ajustes de separación según la altura
  real de cada cuadro, evitando superposiciones y cortes de palabras incómodos.
- Compilación sin avisos de desborde ni referencias pendientes.

El script reproducible de comparación y las imágenes de revisión están en
`.review/fase2_revision/`. La compilación se realiza en su subcarpeta `build/`
y se copia el resultado verificado al PDF principal.
