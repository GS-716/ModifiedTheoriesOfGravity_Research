# Fase 4.4 — Criterios de aceptación

## Representación

- [x] `Variation` es inmutable, indexado y serializable.
- [x] `VolumeElement` distingue `sqrt(-g)` de un escalar ordinario.
- [x] Los objetos calculados no son admitidos como entrada de `ModelSpec`.

## Derivación parcial

- [x] Linealidad, Leibniz, cadena y potencias.
- [x] Derivada respecto de `phi`.
- [x] Derivada tensorial respecto de `g^{ab}`, `R_abcd` y `u_a`.
- [x] Proyección simétrica del momento métrico.
- [x] Proyección completa de Riemann del momento de curvatura.

## Variaciones elementales

- [x] `delta g_ab=-g_ac g_bd delta g^cd`.
- [x] `delta sqrt(-g)=-1/2 sqrt(-g) g_ab delta g^ab`.
- [x] Variación independiente de `R_abcd`.
- [x] Distinción entre `delta u_a` independiente y `nabla_a delta phi`.
- [x] Reconstrucción estructurada de `delta L`.

## Verificación

- [x] Caso cinético escalar con momentos conocidos.
- [x] Potencial `V(phi)` con derivada funcional.
- [x] Momento de curvatura del escalar de Ricci.
- [x] Simetrías y Bianchi algebraico de `P^{abcd}`.
- [x] Serialización de los cuatro momentos.
- [x] Compatibilidad con todas las fases anteriores.

## Próxima tarea recomendada

Restituir la dependencia geométrica de la curvatura, aplicar Palatini e integrar
por partes para obtener `E_ab`, `E_phi` y las contribuciones métricas y escalares
al potencial de frontera `Theta^a`.
