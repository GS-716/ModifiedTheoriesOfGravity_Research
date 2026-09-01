# JSON canónico y vista LaTeX

## Fuente canónica

`results.json` es la salida destinada a otras aplicaciones, notebooks y futuras
etapas. Conserva la estructura de índices, tipos de nodo, trazabilidad y todos
los objetos calculados. `verification.json` ofrece una vista independiente y
compacta de los controles.

## Presentación

`report.tex` traduce la IR a notación matemática mediante dos vistas principales:
`Expresiones tensoriales abstractas` y `Expresiones proyectadas mediante el
ansatz`. Ambas recorren L, M, P, J, F, las ecuaciones de campo y las cantidades
geométricas derivadas en el mismo orden. La primera nunca sustituye el ansatz;
la segunda conserva componentes dispersas y estados por cantidad.

Si `pdflatex` o `xelatex` está disponible, la exportación también genera
`report.pdf`. Una limitación de proyección no elimina la expresión abstracta ni
detiene las demás cantidades.

No se necesita Wolfram Engine para exportar. xAct entra antes, como backend o
fuente de validación; la fase 9 solo registra su evidencia ya obtenida.

La capa DisplayPolicy simplifica únicamente la vista del documento. Sus
expresiones, operaciones e hipótesis se registran en presentation.json y en
comentarios del .tex; results.json sigue siendo la fuente canónica. Véase
[política de presentación](../display-policy.md) para la API, las condiciones
de no nulidad, el inventario del manifiesto y las limitaciones.

Las contracciones tensoriales de Kronecker se realizan en la IR canónica, no
en DisplayPolicy. El historial opcional `delta_contractions` y su archivo de
auditoría se describen en [contracciones delta](../delta-contractions.md).
La cabecera del reporte resume ese historial sin añadir otra sección principal.
