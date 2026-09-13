# Navegación e incorporación de holografía — 12 de septiembre de 2026

Registro histórico de la versión de 154 páginas. Para la versión vigente de 151 páginas, ocho secciones y notación actualizada, consultar `FUSION_REVISION.md` y `../README.md`.

## Versión y respaldo

Título: «Acciones gravitacionales: variación, simetrías y holografía».
Subtítulo: «Lanczos–Lovelock y aplicaciones con campo escalar».
Respaldo previo: `../backups/pre-holografia-20260912/`, con las fuentes y el PDF de 119 páginas.
Resultado: 154 páginas; 140 de exposición y 14 de navegación detallada al final. No hay revelados progresivos.

## Navegación

Todas las páginas tienen un enlace discreto «Contenido» que vuelve a la página 2. Los títulos de sección del contenido general abren los mapas iniciales; «Ver todos los subtemas» abre un índice detallado. Cada entrada y su número de página enlazan directamente con el frame correspondiente. Los índices se distribuyen en páginas equilibradas, con columnas separadas para títulos y números.

El pie de la exposición permite saltar a cualquiera de las siete secciones y conserva el indicador de hitos. Los índices detallados se encuentran después de las referencias para no interrumpir el recorrido expositivo.

Los destinos `nav-frame-*` identifican los frames; `nav-map-*`, los mapas; `nav-index-*`, los índices; y `nav-content`, el contenido general. Las fuentes de navegación están en `contenido/navegacion/`. El script de revisión `.review/holografia/prepare_navigation.py` recorre las fuentes y emite un parche, pero no lo aplica ni se ejecuta durante la compilación habitual. Al cambiar frames deben actualizarse también los índices y verificarse sus destinos.

## Fuente y alcance de la sección nueva

T. Padmanabhan y D. Kothawala, *Lanczos-Lovelock models of gravity*, §2.7, ecuaciones (63)–(78), [arXiv:1302.2151](https://arxiv.org/abs/1302.2151). Se consultó el PDF existente en `_Papers/Lanczos-Lovelock models of gravity.pdf`.

Se insertó una sección completa después de «Condición de segundo orden» y antes de «Generalización con campo escalar». Ocupa las páginas 61–81: 19 frames de explicación y deducción, más mapas de apertura y cierre.

Recorrido:

- Páginas 62–66: alcance, variación mecánica, construcción del término de borde para un dato general, momento fijo, ejemplo explícito y extensión a campos. Corresponde al razonamiento de (63)–(68).
- Páginas 67–72: expansión de la curvatura, integración por partes, separación volumen–superficie, identidad holográfica de Einstein–Hilbert, variables densitizadas, momento gravitacional y variación de frontera. Desarrolla (69)–(73).
- Páginas 73–78: homogeneidad en curvatura, identificación de Q=P/m, condición de divergencia nula, dependencia de segundos gradientes, dos integraciones por partes y deducción de la identidad holográfica LL mediante homogeneidad métrica. Desarrolla (74)–(78), explicitando pasos intermedios.
- Páginas 79–80: comprobación de Einstein–Hilbert, dimensión crítica, suma de órdenes y alcance de lo demostrado.

Se distingue de manera consistente el escalar lagrangiano de su densidad. Las fórmulas nuevas llevan numeración H1–H15, sin modificar la numeración anterior. Se explicita que en D=2m no se puede dividir por D/2−m ni deducir que desaparece el término superficial. La identidad para una suma de órdenes se presenta con sus pesos, no con un único orden efectivo.

La holografía tratada es la relación entre las partes de volumen y superficie de la acción, no una afirmación de dualidad AdS/CFT. No se ha extendido esta propiedad al campo escalar. El material escalar ya existente se conserva como parte posterior del Beamer.

## Validación

- Compilación completa de 154 páginas, sin desbordes, destinos duplicados ni referencias indefinidas.
- 1.566 enlaces internos con destinos resueltos; retorno a «Contenido» en todas las páginas.
- Los 138 frames de contenido y mapas, páginas 3–140, están cubiertos por los índices detallados.
- Se conservaron los 62 rótulos de ecuaciones originales y sus números.
- Comparación textual del contenido previo y comparación visual de sus 117 páginas de desarrollo: sin cambios fuera del pie de navegación.
- Revisión visual de portada, contenido, sección nueva y todos los índices; ajuste de mapas, columnas de números y distribución de entradas.
- Comprobación numérica auxiliar de las identidades de las variables densitizadas y del momento de Einstein–Hilbert en dimensiones 3, 4 y 5; no sustituye la deducción incluida en los frames.

Evidencias de trabajo: `.review/holografia/audit.json`, `audit.py`, `check_math.py`, `pages/` y hojas de revisión `qa-*.png`.
