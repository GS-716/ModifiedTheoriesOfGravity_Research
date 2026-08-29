# Fase 2.2 — Canonización y simetrías

## Forma canónica estructural

La canonización realiza, en orden controlado:

1. expansión distributiva;
2. higiene de índices mudos;
3. aplicación de simetrías monótérmino;
4. normalización de coeficientes racionales;
5. ordenamiento determinista de factores;
6. combinación de términos estructuralmente iguales;
7. nueva canonización de índices mudos.

El resultado es idempotente: canonizar dos veces no cambia la expresión.

## Simetrías admitidas

Para la métrica se aplica

\[
g^{ab}=g^{ba}
\]

y, para un tensor completamente covariante o contravariante con simetría de
Riemann,

\[
R_{abcd}=-R_{bacd}=-R_{abdc}=R_{cdab}.
\]

Las simetrías que relacionan varios monomios por una suma, como

\[
R_{abcd}+R_{acdb}+R_{adbc}=0,
\]

no se reducen automáticamente en este backend. El núcleo construye y conserva
el residual de Bianchi con estado `undetermined`. Un backend futuro con soporte
multitémino, como xAct, deberá decidirlo.

## Identidades dimensionales

No se aplican identidades dependientes de una dimensión concreta, salvo la
traza elemental `delta^a_a = D`. Identidades de Schouten, dualizaciones y
reducciones especiales en `D=3` o `D=4` requieren una capacidad separada.

