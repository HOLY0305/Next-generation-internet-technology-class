$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
  function Invoke-Tool([string]$Tool, [string[]]$Arguments) {
    & $Tool @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Tool failed with exit code $LASTEXITCODE" }
  }
  Invoke-Tool 'xelatex' @('-interaction=nonstopmode','-halt-on-error','main.tex')

  $aux = if (Test-Path 'main.aux') { Get-Content 'main.aux' -Raw } else { '' }
  if ($aux.Contains('\bibdata{') -and $aux.Contains('\citation{')) {
    Invoke-Tool 'bibtex' @('main')
  } else {
    if (Test-Path 'main.bbl') { Remove-Item -LiteralPath 'main.bbl' }
    Write-Host 'No active bibliography citations; skipped BibTeX.'
  }

  Invoke-Tool 'xelatex' @('-interaction=nonstopmode','-halt-on-error','main.tex')
  Invoke-Tool 'xelatex' @('-interaction=nonstopmode','-halt-on-error','main.tex')
} finally { Pop-Location }
