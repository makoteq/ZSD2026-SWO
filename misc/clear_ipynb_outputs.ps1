param(
    [string]$RootPath = "."
)

$root = Resolve-Path $RootPath
Write-Host "Searching for .ipynb files under $root"

$files = Get-ChildItem -Path $root -Recurse -Filter *.ipynb -File
if (-not $files) {
    Write-Host "No .ipynb files found."
    exit 0
}

$hasNbconvert = Get-Command jupyter -ErrorAction SilentlyContinue
$pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
} else {
    $null
}

foreach ($file in $files) {
    Write-Host "Clearing: $($file.FullName)"
    if ($hasNbconvert) {
        jupyter nbconvert --clear-output --inplace $file.FullName
    } elseif ($pythonCmd) {
        $pyCode = @"
import json
from pathlib import Path

path = Path(r'''$($file.FullName)''')
text = path.read_text(encoding='utf-8')
data = json.loads(text)
changed = False
for cell in data.get('cells', []):
    if cell.get('cell_type') == 'code':
        if cell.get('outputs'):
            cell['outputs'] = []
            changed = True
        if cell.get('execution_count') is not None:
            cell['execution_count'] = None
            changed = True
if changed:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
"@
        & $pythonCmd -c $pyCode
    } else {
        Write-Host "Neither jupyter nor python were found on PATH. Install Python and/or nbconvert."
        exit 1
    }
}

Write-Host "Done."
