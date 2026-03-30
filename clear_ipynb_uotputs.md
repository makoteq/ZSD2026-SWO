
# Running PowerShell Script

## Command Breakdown

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\clear_ipynb_outputs.ps1 -RootPath .
```

| Parameter | Purpose |
|-----------|---------|
| `-NoProfile` | Skips loading PowerShell profile |
| `-ExecutionPolicy Bypass` | Allows script execution without restrictions |
| `-File` | Specifies script file to run |
| `.\clear_ipynb_outputs.ps1` | Script name (current directory) |
| `-RootPath .` | Parameter passed to script (current directory) |

## How to Run

1. Open Command Prompt or PowerShell
2. Navigate to your script directory
3. Execute the command above

## Alternative (Direct PowerShell)

```powershell
.\clear_ipynb_outputs.ps1 -RootPath .
```

Note: May require execution policy adjustment first.
