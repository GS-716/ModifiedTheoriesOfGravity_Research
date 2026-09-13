# Compactación del Beamer, 12 de septiembre de 2026

## Resultado

151 → 119 diapositivas: 32 menos (21,2 %). Todas son estáticas.

Se revisó la presentación completa antes de elegir las fusiones. Se priorizó retirar páginas de orientación repetidas, ya que gran parte de las deducciones ocupa casi toda la altura útil. No se trasladaron demostraciones a un apéndice ni se suprimieron ecuaciones para alcanzar una cuota de páginas.

| Cambio | Reducción |
| --- | ---: |
| 25 mapas intermedios se convierten en actualizaciones del contador del pie | 25 |
| Los objetivos de las seis portadas se integran en sus mapas de apertura; el índice general permanece | 6 |
| Diagnóstico de cuarto orden y condición de divergencia nula se presentan en dos columnas | 1 |
| Total | 32 |

## Orientación

Permanecen los seis mapas completos al inicio y al cierre de cada sección: 12 páginas. Todos sus nodos y descripciones se conservaron literalmente, incluidos Bianchi, Noether y los casos de control. Los objetivos de sección se conservaron literalmente.

Los hitos intermedios ya no generan una página repetida. El pie muestra `k/n` junto a la sección activa: número de bloques terminados. Por ejemplo, `Acción 3/6` indica que ya se desarrollaron los tres primeros bloques. El contador cambia en los mismos límites anteriores. En segundo orden, los bloques 1 y 2 se desarrollan en una misma página; el contador pasa de 0/3 a 2/3 al concluirla.

Este cambio modifica deliberadamente la representación de los hitos solicitada en la fase 2 para compactar: no reaparece el diagrama completo tras cada bloque. La información del mapa permanece en las aperturas y cierres.

## Fusión teórica

Las antiguas páginas 71 y 73 forman la nueva página 58. La columna izquierda conserva íntegramente el diagnóstico de derivadas superiores; la derecha conserva íntegramente la condición, su consecuencia para el tensor de campo y el acoplamiento a materia. Solo se cambian el título, los encabezados de columna y la disposición.

Se utiliza `footnotesize` (9 pt) en esta página, frente a `small` (10 pt) en las dos anteriores. No se aplicó una reducción global. Las restantes deducciones conservan su tamaño y disposición.

## Comprobaciones

- Comparación exacta del cuerpo principal después de normalizar únicamente títulos, comentarios, espacios, rutas y comandos de presentación explícitamente modificados: no cambió el texto teórico ni las fórmulas.
- Las 62 etiquetas de ecuación y sus números se conservaron.
- Todas las fuentes de tablas y del cierre son idénticas, byte por byte, a las del respaldo.
- Las seis definiciones de contenido de los mapas son idénticas al respaldo.
- 112 páginas conservadas son idénticas píxel por píxel al PDF anterior, excluyendo el pie, cuya numeración y contador cambiaron.
- Se renderizó el PDF completo y se revisaron las siete páginas con redistribución de contenido: seis aperturas y la fusión.
- Se corrigieron los desbordes detectados en la primera compilación. La compilación final no registra cajas desbordadas ni referencias indefinidas.

Auditoría reproducible: `.review/compactacion/audit.py`; correspondencia entre páginas antiguas y nuevas: `.review/compactacion/audit.json`. Estos controles verifican preservación respecto del documento previo, no constituyen una validación científica independiente.

## Respaldo y recuperación

El estado de 151 páginas se encuentra en `backups/pre-compactacion-20260912/`, con el PDF, las fuentes completas y los registros anteriores. SHA-256 del PDF respaldado:

`DAB5E0621AF9C8258CBEE6E04F563AA4BD2B93DE040FBE4FBBDDB1DBAF605ADF`

Se puede abrir o compilar ese respaldo en su propia carpeta. La estructura anterior se conserva dentro de él; no es necesario deshacer la organización nueva para consultarlo.
