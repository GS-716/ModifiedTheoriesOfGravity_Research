# Outputs

Destino de resultados generados. Estos archivos serán artefactos reproducibles,
no fuente canónica del motor.

`phase5_wolfram_validation.json` se regenera desde la raíz con:

```powershell
python scripts/validate_phase5_wolfram.py
```

`phase6_wolfram_validation.json` conserva la validación Noether–Wald y se
regenera con:

```powershell
python scripts/validate_phase6_wolfram.py
```

`phase7_wolfram_validation.json` conserva las ocho comprobaciones coordenadas
de FLRW realizadas con xCoba y se regenera con:

```powershell
python scripts/validate_phase7_wolfram.py
```

`phase8_reference_verification.json` reúne los controles internos del modelo
escalar–tensor general y se regenera con:

```powershell
python scripts/verify_phase8_reference.py
```

La carpeta `phase9_reference` contiene una exportación reconstruible, su
manifiesto de integridad y la vista LaTeX. Se regenera con:

```powershell
python scripts/export_phase9_reference.py
```

`phase10_reference` demuestra la API integral, conserva el `ModelSpec` de
entrada y genera el bundle final. Se regenera con:

```powershell
python scripts/run_phase10_reference.py
```

`phase11_reference` se genera desde la CLI integral con `--wolfram`. Un reporte
genérico independiente puede regenerarse para cualquier ModelSpec JSON con:

```powershell
python scripts/validate_phase11_wolfram.py ruta/al/model.json
```

`phase12_reference` demuestra la validación diferencial y la adjudicación
trazable dentro de una corrida integral. Se regenera con:

```powershell
python scripts/run_phase12_reference.py
```

`phase13_reference` contiene la especificación y el reporte comparativo de los
cinco modelos incorporados, además de un bundle por modelo. Se regenera con:

```powershell
python scripts/run_phase13_catalog.py
```

`phase14_reference` conserva una fuente textual compuesta, el `ModelSpec`
compilado y su bundle validado con xAct. Se regenera con:

```powershell
python scripts/run_phase14_source.py
```
