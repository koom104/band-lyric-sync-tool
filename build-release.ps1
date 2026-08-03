param(
    [string]$Version = "0.1.0",
    [switch]$SkipDependencies,
    [switch]$SkipArchive,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildRoot = Join-Path $Root "build"
$PortableRoot = Join-Path $BuildRoot "BandLyricSync-portable"
$RuntimeRoot = Join-Path $PortableRoot "runtime"
$SitePackages = Join-Path $RuntimeRoot "Lib\site-packages"
$ReleaseRoot = Join-Path $Root "release"
$CacheRoot = Join-Path $BuildRoot "cache"
$DevPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $DevPython)) {
    throw "Development virtual environment was not found: $DevPython"
}

New-Item -ItemType Directory -Force -Path $BuildRoot, $ReleaseRoot, $CacheRoot | Out-Null

if (-not $SkipDependencies) {
    if (Test-Path -LiteralPath $PortableRoot) {
        $resolvedBuild = (Resolve-Path -LiteralPath $BuildRoot).Path
        $resolvedPortable = (Resolve-Path -LiteralPath $PortableRoot).Path
        if (-not $resolvedPortable.StartsWith($resolvedBuild, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove a directory outside the build root: $resolvedPortable"
        }
        Remove-Item -LiteralPath $PortableRoot -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $RuntimeRoot, $SitePackages | Out-Null
    $PythonArchive = Join-Path $CacheRoot "python-3.11.9-embed-amd64.zip"
    if (-not (Test-Path -LiteralPath $PythonArchive)) {
        Invoke-WebRequest `
            -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" `
            -OutFile $PythonArchive
    }
    Expand-Archive -LiteralPath $PythonArchive -DestinationPath $RuntimeRoot -Force

    $PathConfig = Join-Path $RuntimeRoot "python311._pth"
    $PathLines = Get-Content -LiteralPath $PathConfig
    $PathLines = $PathLines -replace "^#import site$", "import site"
    Set-Content -LiteralPath $PathConfig -Value $PathLines -Encoding ASCII

    & $DevPython -m pip install `
        --disable-pip-version-check `
        --upgrade `
        --target $SitePackages `
        pip `
        -r (Join-Path $Root "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }

    # PySoundFile is a legacy Sync Toolbox dependency that installs the same
    # soundfile.py module. Reinstall the pinned modern package last.
    & $DevPython -m pip install `
        --disable-pip-version-check `
        --upgrade `
        --force-reinstall `
        --no-deps `
        --target $SitePackages `
        "soundfile==0.12.1"
    if ($LASTEXITCODE -ne 0) {
        throw "SoundFile compatibility repair failed."
    }
}

foreach ($PrefixDirectory in @("Library", "share")) {
    $TargetInstalledPath = Join-Path $SitePackages $PrefixDirectory
    $RuntimePrefixPath = Join-Path $RuntimeRoot $PrefixDirectory
    if (Test-Path -LiteralPath $TargetInstalledPath) {
        if (Test-Path -LiteralPath $RuntimePrefixPath) {
            Remove-Item -LiteralPath $RuntimePrefixPath -Recurse -Force
        }
        Move-Item -LiteralPath $TargetInstalledPath -Destination $RuntimePrefixPath
    }
}

foreach ($RuntimeDll in @("msvcp140.dll", "vcomp140.dll")) {
    $RuntimeDllSource = Join-Path "${env:WINDIR}\System32" $RuntimeDll
    if (Test-Path -LiteralPath $RuntimeDllSource) {
        Copy-Item -LiteralPath $RuntimeDllSource -Destination (Join-Path $RuntimeRoot $RuntimeDll) -Force
    }
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $PortableRoot "app"), `
    (Join-Path $PortableRoot "bin") | Out-Null

Copy-Item -LiteralPath (Join-Path $Root "app.py") -Destination (Join-Path $PortableRoot "app\app.py") -Force
Copy-Item -LiteralPath (Join-Path $Root "packaging\launcher.py") -Destination (Join-Path $PortableRoot "app\launcher.py") -Force
Copy-Item -LiteralPath (Join-Path $Root "README.md") -Destination (Join-Path $PortableRoot "README.md") -Force
Copy-Item -LiteralPath (Join-Path $Root "THIRD_PARTY_NOTICES.txt") -Destination (Join-Path $PortableRoot "THIRD_PARTY_NOTICES.txt") -Force

foreach ($ToolName in @("ffmpeg", "ffprobe")) {
    $Command = Get-Command $ToolName -ErrorAction Stop
    $CommandItem = Get-Item -LiteralPath $Command.Source
    $ToolSource = if ($CommandItem.Target) { $CommandItem.Target } else { $CommandItem.FullName }
    Copy-Item -LiteralPath $ToolSource -Destination (Join-Path $PortableRoot "bin\$ToolName.exe") -Force
}

$LauncherOutput = Join-Path $PortableRoot "BandLyricSync.exe"
if (Test-Path -LiteralPath $LauncherOutput) {
    Remove-Item -LiteralPath $LauncherOutput -Force
}
$Csc = "${env:WINDIR}\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path -LiteralPath $Csc)) {
    throw "Windows C# compiler was not found: $Csc"
}
& $Csc `
    /nologo `
    /target:winexe `
    /optimize+ `
    "/out:$LauncherOutput" `
    /reference:System.dll `
    /reference:System.Windows.Forms.dll `
    (Join-Path $Root "packaging\Launcher.cs")
if ($LASTEXITCODE -ne 0) {
    throw "Native launcher compilation failed."
}

@"
@echo off
start "" "%~dp0runtime\pythonw.exe" "%~dp0app\launcher.py"
"@ | Set-Content -LiteralPath (Join-Path $PortableRoot "Launch Band Lyric Sync.cmd") -Encoding ASCII

@"
Band Lyric Sync $Version

실행:
  BandLyricSync.exe

사용자 데이터:
  %LOCALAPPDATA%\BandLyricSync

GPU:
  NVIDIA 드라이버가 설치된 PC에서는 CUDA를 자동 사용합니다.
  GPU를 사용할 수 없으면 CPU로 실행됩니다.

모델:
  Whisper 및 Demucs 모델은 최초 사용 시 다운로드됩니다.
"@ | Set-Content -LiteralPath (Join-Path $PortableRoot "DISTRIBUTION.txt") -Encoding UTF8

$env:PATH = (Join-Path $PortableRoot "bin") + [IO.Path]::PathSeparator + $env:PATH
$env:BAND_LYRIC_SYNC_DATA_DIR = Join-Path $BuildRoot "validation-data"
& (Join-Path $RuntimeRoot "python.exe") -c "import gradio, torch, demucs, stable_whisper, synctoolbox, soundfile; assert hasattr(soundfile, 'SoundFileRuntimeError'); print('runtime-ok', torch.__version__, torch.cuda.is_available(), soundfile.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Portable runtime import validation failed."
}
& (Join-Path $RuntimeRoot "python.exe") -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "Portable runtime dependency validation failed."
}

$PortableArchive = Join-Path $ReleaseRoot "BandLyricSync-$Version-portable-win64.zip"
if (-not $SkipArchive) {
    if (Test-Path -LiteralPath $PortableArchive) {
        Remove-Item -LiteralPath $PortableArchive -Force
    }
    & tar.exe -a -c -f $PortableArchive -C $BuildRoot "BandLyricSync-portable"
    if ($LASTEXITCODE -ne 0) {
        throw "Portable archive creation failed."
    }
}

if (-not $SkipInstaller) {
    $Iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if (-not $Iscc) {
        $DefaultIscc = "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe"
        if (Test-Path -LiteralPath $DefaultIscc) {
            $Iscc = Get-Item -LiteralPath $DefaultIscc
        }
    }
    if (-not $Iscc) {
        throw "Inno Setup compiler (ISCC.exe) was not found."
    }
    $IsccPath = if ($Iscc.Source) { $Iscc.Source } else { $Iscc.FullName }
    & $IsccPath `
        "/DAppVersion=$Version" `
        "/DSourceRoot=$PortableRoot" `
        "/DOutputRoot=$ReleaseRoot" `
        (Join-Path $Root "packaging\BandLyricSync.iss")
    if ($LASTEXITCODE -ne 0) {
        throw "Installer creation failed."
    }
}

$Artifacts = Get-ChildItem -LiteralPath $ReleaseRoot -File | Where-Object {
    $_.Name -eq "BandLyricSync-Setup-$Version-win64.exe" -or
    $_.Name -eq "BandLyricSync-$Version-portable-win64.zip"
} | ForEach-Object {
    [PSCustomObject]@{
        Name = $_.Name
        Size = $_.Length
        Sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
}
$Artifacts | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ReleaseRoot "release-manifest.json") -Encoding UTF8
$Artifacts | Format-Table -AutoSize
