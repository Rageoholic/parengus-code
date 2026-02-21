if (-not (Test-Path "out\samp-app\debug")) {
    mkdir "out\samp-app\debug" | Out-Null
}

$srcExe = ".\target\debug\samp-app.exe"
$dstExe = "out\samp-app\debug\samp-app.exe"
$srcPdb = ".\target\debug\samp_app.pdb"
$dstPdb = "out\samp-app\debug\samp_app.pdb"

if ((Test-Path $dstExe) -and ((Get-Item $srcExe).LastWriteTime -le (Get-Item $dstExe).LastWriteTime)) {
    Write-Host "Up-to-date: samp-app.exe + samp_app.pdb"
} else {
    Write-Host "Copying samp-app.exe + samp_app.pdb..."
    Copy-Item $srcExe $dstExe
    Copy-Item $srcPdb $dstPdb
}
