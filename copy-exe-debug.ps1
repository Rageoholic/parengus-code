
if (-not (Test-Path "game-debug")) {
    mkdir "game-debug"
}


Copy-Item ".\target\debug\samp-app.exe" "game-debug\samp-app.exe"
Copy-Item ".\target\debug\samp_app.pdb" "game-debug\samp_app.pdb"