$ErrorActionPreference = 'Stop'
$beamerRoot = $PSScriptRoot
$buildPath = Join-Path $beamerRoot '.build'
New-Item -ItemType Directory -Path $buildPath -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $beamerRoot 'main.tex') -Destination (Join-Path $buildPath 'main.tex') -Force
Copy-Item -LiteralPath (Join-Path $beamerRoot 'contenido') -Destination $buildPath -Recurse -Force
Push-Location $buildPath
try {
    foreach ($passNumber in 1..2) {
        & pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "Error de LaTeX en la pasada $passNumber. Consulte .build/main.log." }
    }
    $buildLog = Get-Content -LiteralPath (Join-Path $buildPath 'main.log') -Raw
    if ($buildLog -match 'Overfull|LaTeX Warning:.*undefined|There were undefined|Rerun to get') {
        throw 'El PDF no se publica: revise los avisos de .build/main.log.'
    }
    Copy-Item -LiteralPath (Join-Path $buildPath 'main.pdf') -Destination (Join-Path $beamerRoot 'main.pdf') -Force
    Write-Host 'PDF actualizado. Revise visualmente las paginas modificadas antes de presentar.'
}
finally {
    Pop-Location
}
