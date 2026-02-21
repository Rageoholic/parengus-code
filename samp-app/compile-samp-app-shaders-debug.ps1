$sourceDir = "samp-app\shaders"
$outputDir = "out\samp-app\debug\shaders"

if (-not (Test-Path $outputDir)) {
    mkdir $outputDir | Out-Null
}

$compiled = 0
$skipped = 0

Get-ChildItem "$sourceDir\*.slang" | ForEach-Object {
    $src = $_
    $dst = Join-Path $outputDir ($src.BaseName + ".spv")

    if ((Test-Path $dst) -and ($src.LastWriteTime -le (Get-Item $dst).LastWriteTime)) {
        $skipped++
        return
    }

    Write-Host "Compiling $($src.Name)..."
    slangc $src.FullName -target spirv -o $dst
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to compile $($src.Name)"
        exit 1
    }
    $compiled++
}

Write-Host "Shaders: $compiled compiled, $skipped up-to-date"
