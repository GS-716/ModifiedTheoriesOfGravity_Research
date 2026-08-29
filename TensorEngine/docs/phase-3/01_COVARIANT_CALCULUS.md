# Fase 3.1 — Cálculo covariante formal

La fase 3 introduce derivadas covariantes abstractas sobre la IR. No calcula
símbolos de Christoffel ni componentes coordenadas.

## Reglas implementadas

El operador `covariant_derivative` aplica:

- linealidad;
- regla de Leibniz;
- regla de la cadena para funciones simbólicas;
- derivación de potencias con exponente racional constante;
- compatibilidad métrica `∇_a g_bc = 0`;
- constancia covariante del delta de Kronecker;
- constancia de parámetros y de la dimensión simbólica;
- identificación `∇_a phi = u_a`;
- simetría del Hessiano de `phi` para conexión sin torsión.

Los tensores generales, `Riemann` y derivadas posteriores de `u_a` permanecen
como nodos `CovariantDerivative`, preservando sus índices y su procedencia.

## Derivadas de funciones

Una función declarada `F(phi)` se deriva como

\[
\nabla_a F(\phi)=F_{,\phi}(\phi)u_a.
\]

`FunctionDerivative` almacena el nombre de la función, un orden por argumento y
los argumentos originales. Esto permite representar sin ambigüedad derivadas
mixtas y de orden superior y serializarlas a JSON.

## Límites deliberados

Una potencia con exponente simbólico general permanece como derivada formal. No
se introducen automáticamente logaritmos ni supuestos de dominio que el modelo
no haya declarado.

