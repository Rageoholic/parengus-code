[CmdletBinding()]
param(
    [switch]$ShaderDebugInfo
)

$sourceDir = "samp-app\shaders"
$outputDir = "out\samp-app\debug\shaders"

if (-not (Test-Path $outputDir)) {
    mkdir $outputDir | Out-Null
}

if ($ShaderDebugInfo) {
    Write-Host "Note: -ShaderDebugInfo is now unnecessary;"
    Write-Host "both shader variants are always compiled."
}

$compiled = 0
$skipped = 0

$shaderVariants = @(
    @{
        Suffix    = ""
        ExtraArgs = @()
    },
    @{
        Suffix    = ".debug"
        ExtraArgs = @("-g")
    }
)

Get-ChildItem "$sourceDir\*.slang" | ForEach-Object {
    $src = $_

    foreach ($variant in $shaderVariants) {
        $dst = Join-Path $outputDir (
            $src.BaseName + $variant.Suffix + ".spv"
        )

        if (
            (Test-Path $dst) -and
            ($src.LastWriteTime -le (Get-Item $dst).LastWriteTime)
        ) {
            $skipped++
            continue
        }

        Write-Host "Compiling $($src.Name) -> $(Split-Path $dst -Leaf)..."

        $slangcArgs = @(
            $src.FullName
            "-target"
            "spirv"
            "-o"
            $dst
        ) + $variant.ExtraArgs

        slangc @slangcArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to compile $($src.Name)"
            exit 1
        }

        $compiled++
    }
}

Write-Host "Shaders: $compiled compiled, $skipped up-to-date"
