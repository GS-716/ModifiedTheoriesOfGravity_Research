# Beamer: teorías de gravedad modificada

La versión unificada vigente es `main.pdf`: «Teorías de gravedad modificada», con subtítulo «Variación, simetrías y estructura holográfica: de Lanczos–Lovelock al acoplamiento escalar EQT». Contiene 151 páginas estáticas: 138 de presentación y 13 de índice detallado al final. Integra la ampliación de `Beamer_holografia_generalizada_SISTEMA`. Se edita en `main.tex` y en las fuentes de `contenido/`.

## Organización

- `contenido/mapas/`: las ocho secciones tienen mapa completo de apertura y una fila de casillas que se colorea durante el desarrollo. Cada capítulo termina con un resumen de resultados. No hay overlays ni flechas.
- `contenido/holografia/ll-compacto.tex`: sección LL, páginas 60–69, con deducciones H1–H15 en 10 frames, incluido el mapa y el resumen.
- `contenido/holografia/generalizada.tex`: generalización revisada, páginas 118–127, con deducciones G1–G10 en 10 frames, incluido el mapa y el resumen.
- `contenido/navegacion/`: contenido general e índices enlazados a cada diapositiva.
- `contenido/tablas/`: tablas comparativas y diccionario de momentos.
- `contenido/cierre/`: gráfico radial y síntesis final.
- `documentacion/COMPACTACION.md`: criterio aplicado y comprobaciones.
- `documentacion/NAVEGACION_HOLOGRAFIA.md`: registro histórico de la primera incorporación de holografía, anterior a la fusión.
- `documentacion/FUSION_REVISION.md`: fusión actual, correcciones teóricas y simplificaciones. Los registros anteriores describen versiones históricas.
- `documentacion/historial/`: registros de las fases anteriores. Sus rutas describen la organización que existía entonces.
- `backups/`: versiones anteriores, sin modificar. `pre-fusion-20260912/base/` conserva la versión de 154 páginas y `pre-fusion-20260912/sistema/` la de 165 páginas. La fecha del nombre corresponde al inicio del trabajo. Se conservan los respaldos históricos anteriores.
- `archivo/`: primeros mapas sustituidos, fase 4 retirada y auxiliares antiguos. Incluye el antiguo `main-lectura.pdf`, idéntico a la versión de 151 páginas. No es un PDF vigente.
- `.review/`: compilaciones, renders y auditorías de cada revisión. No son archivos para presentar.

La carpeta duplicada `Beamer_holografia_generalizada_SISTEMA` se retira de la raíz tras comprobar que todos sus archivos coinciden con el respaldo `pre-fusion-20260912/sistema/`. El antiguo desarrollo LL no utilizado queda recuperable en el respaldo `base/`. `pre-mapas-resumenes-20260913/` conserva además el estado de trabajo anterior a esta revisión (incluido el PDF que estaba publicado entonces). No se borran los respaldos históricos.

## Compilar

Con MiKTeX y `pdflatex` disponibles, ejecutar desde PowerShell:

```powershell
& 'C:\Investigacion\_Beamer\compilar.ps1'
```

El script copia las fuentes activas a `.build/`, compila dos veces y actualiza únicamente `main.pdf`. Interrumpe la publicación si encuentra errores, referencias indefinidas o desbordes. Tras editar contenido, revisar también visualmente el PDF: una compilación limpia no garantiza por sí sola un buen diseño.

El botón «Contenido» del pie está disponible en todas las páginas. Desde el contenido general se puede abrir cada sección o su índice completo. Si se añaden, eliminan o renombran frames, actualizar también `contenido/navegacion/indices.tex` y conservar destinos únicos; compilar no regenera por sí solo ese índice.

## Criterios que se deben conservar

Mantener las deducciones completas, la numeración de ecuaciones, las comprobaciones de Bianchi y Noether y el detalle de los casos. No añadir revelados progresivos. Los mapas no llevan flechas. Evitar reducir globalmente el tamaño del texto.

Notación: `L` es el lagrangiano escalar, `\mathscr L` su densidad, `\ell` la escala de longitud de EQT, `\theta` la coordenada angular y `\phi` el campo escalar. El operador de escalamiento se escribe `\mathcal H_g`. Las convenciones y la distinción de los momentos métricos se explican en la página 5.

Las expresiones de holografía generalizada son deducciones del formalismo del Beamer, no resultados atribuidos al artículo LL.

## Verificar y mantener la navegación

Desde esta carpeta, con Python disponible:

```powershell
python -m pip install -r herramientas/requirements.txt
python herramientas/generar_indice.py --check
python herramientas/verificar_pdf.py main.pdf
```

Si cambian los frames, ejecutar `python herramientas/generar_indice.py --write`, revisar los cambios y volver a compilar. El generador conserva los identificadores existentes; sin opciones emite un parche. La auditoría comprueba destinos internos, acceso a Contenido, cobertura del índice, mapas, resúmenes y el límite de los capítulos holográficos. No sustituye la revisión visual.

El repositorio incluye las fuentes activas, el PDF publicado, las herramientas y la documentación. `backups/`, `archivo/`, `.review/`, `.build/` y los auxiliares son archivos locales excluidos de Git. Los respaldos mencionados arriba permanecen en el equipo de trabajo, no forman parte de una clonación nueva.
