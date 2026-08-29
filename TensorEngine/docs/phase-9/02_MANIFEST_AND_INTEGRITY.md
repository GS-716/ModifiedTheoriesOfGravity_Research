# Manifiesto e integridad

`manifest.json` es el punto de entrada de una exportación. Registra:

- esquema de exportación, ID, fecha UTC y estado final;
- modelo, esquema de modelo, convención y dimensión;
- backend principal y fuentes externas con sus versiones;
- duración total y duración opcional por etapa;
- inventario de expresiones con forma y SHA-256;
- inventario de archivos con tipo de medio, tamaño y SHA-256.

El manifiesto no intenta incluir su propio hash, porque eso crearía una
referencia circular. Sí autentica los tres artefactos derivados:
`results.json`, `verification.json` y `report.tex`.

Las escrituras usan un archivo temporal en el mismo directorio y una
sustitución atómica. Una interrupción no debe dejar un artefacto parcialmente
escrito con el nombre definitivo.

`RunManifest.verify_files(directorio)` vuelve a calcular tamaño y SHA-256,
comprueba que ninguna ruta escape del bundle y devuelve cualquier incidencia.
