# JSON canónico y vista LaTeX

## Fuente canónica

`results.json` es la salida destinada a otras aplicaciones, notebooks y futuras
etapas. Conserva la estructura de índices, tipos de nodo, trazabilidad y todos
los objetos calculados. `verification.json` ofrece una vista independiente y
compacta de los controles.

## Presentación

`report.tex` traduce todos los nodos de la IR a notación matemática y organiza
lagrangiano, momentos, variación, ecuaciones, frontera y resultados opcionales.
El impresor preserva la posición y varianza de cada índice. El documento puede
compilarse con una distribución LaTeX, pero esa compilación no es requisito del
motor ni interviene en cálculos posteriores.

No se necesita Wolfram Engine para exportar. xAct entra antes, como backend o
fuente de validación; la fase 9 solo registra su evidencia ya obtenida.
