# Manifiesto e integridad

`manifest.json` es el punto de entrada de una exportación. Registra:

- esquema de exportación, ID, fecha UTC y estado final;
- modelo, esquema de modelo, convención y dimensión;
- backend principal y fuentes externas con sus versiones;
- duración total y duración opcional por etapa;
- inventario de expresiones con forma y SHA-256;
- inventario de las vistas abstracta y proyectada, con estado y SHA-256;
- inventario de archivos con tipo de medio, tamaño y SHA-256.

El manifiesto no intenta incluir su propio hash, porque eso crearía una
referencia circular. Sí autentica los artefactos derivados:
`results.json`, `verification.json`, `report.tex` y, cuando el compilador está
disponible, `report.pdf`.

Incluye además `presentation.json` y, cuando hay historial de contracciones,
`delta_contractions.json`. Este último registra pasadas, sustituciones, motivos
y conteos finales de deltas. El mismo historial está en el campo opcional de
`results.json`. No es parte del ID semántico, pero sí del hash de ambos archivos.

Las escrituras usan un archivo temporal en el mismo directorio y una
sustitución atómica. Una interrupción no debe dejar un artefacto parcialmente
escrito con el nombre definitivo.

`RunManifest.verify_files(directorio)` vuelve a calcular tamaño y SHA-256,
comprueba que ninguna ruta escape del bundle y devuelve cualquier incidencia.
