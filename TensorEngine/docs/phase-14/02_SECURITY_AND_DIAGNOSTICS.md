# Seguridad y diagnósticos

La fuente se analiza con el árbol sintáctico de Python, pero nunca se ejecuta.
El compilador recorre una lista cerrada de nodos. Se rechazan, entre otros:

- atributos, subscripts, listas, diccionarios, lambdas y comprensiones;
- imports y llamadas indirectas;
- funciones no declaradas, keywords y aridades incorrectas;
- símbolos desconocidos y redefinición de los alias y constructores reservados;
- números decimales, para evitar aproximaciones no declaradas;
- expresiones mayores de 10 000 caracteres o 1 000 nodos.

Cada error incluye línea y columna. La normalización global usa una gramática aún
más estricta: solo enteros, parámetros, dimensión y operaciones algebraicas.

Las cadenas solo se admiten como índices literales de `Riemann`, `metric` y
`gradient`. `contract` valida el resultado con la IR y exige contracción escalar
completa. Los constructores de un registro personalizado son código Python de
confianza proporcionado por el llamador; nunca se deserializan desde la fuente.
