# Native Windows installation with selectable dependency groups.
param(
    [switch]$NewEnv,
    [switch]$Basic,
    [switch]$PyTorch3D,
    [switch]$FlexGemm,
    [switch]$DiffOctreeRast,
    [switch]$NvDiffRast,
    [switch]$CuMesh,
    [switch]$OVoxel,
    [switch]$MapAnything,
    [switch]$NvDiffRec,
    [string]$BuildRoot = (Join-Path $env:TEMP 'plsr-build'),
    [string]$Vcvarsall = '',
    [ValidateRange(1, 2147483647)][int]$MaxJobs = 4,
    [switch]$Help
)

if ($Help -or -not ($NewEnv -or $Basic -or $PyTorch3D -or $FlexGemm -or
        $DiffOctreeRast -or $NvDiffRast -or $CuMesh -or $OVoxel -or $MapAnything -or $NvDiffRec)) {
    Write-Output @'
Usage: .\setup.ps1 [options]
  -NewEnv          Create and activate Conda environment plsr (Python 3.12, PyTorch cu128)
  -Basic           Install requirements.txt, build tools, Triton Windows and CUDA Toolkit
  -PyTorch3D       Install PyTorch3D with its Windows CUDA patch
  -FlexGemm        Install FlexGEMM with its Windows patch
  -DiffOctreeRast  Install the patch-visibility rasterizer
  -NvDiffRast      Install nvdiffrast
  -CuMesh          Install CuMesh with its cubvh Windows patch
  -OVoxel          Install the bundled O-Voxel source
  -MapAnything    Install camera-estimation dependencies
  -NvDiffRec      Install camera-anchor rendering dependencies
  -BuildRoot PATH Source build directory (use a short writable Windows path)
  -Vcvarsall PATH Visual Studio vcvarsall.bat, if auto-detection is unsuitable
  -MaxJobs N      Parallel compiler jobs (default: 4)
  -Help           Show this help
Options can be combined. Without -NewEnv, activate your Conda environment first.
Windows uses PyTorch SDPA; no FlashAttention installation is needed.
'@
    return
}

Write-Warning "This Windows installer was automatically adapted by AI and has only been tested in the author's own environment. Please review it before use."
$answer = Read-Host 'Type y to continue (anything else cancels)'
if ($answer.Trim() -ine 'y') {
    Write-Output 'Installation cancelled.'
    return
}

$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Use setup.sh on Linux/WSL.' }
$repo = $PSScriptRoot
$torchVersion = '2.9.1'
$torchvisionVersion = '0.24.1'
$tritonVersion = '3.5.1.post24'

function Assert-LastExit([string]$description) {
    if ($LASTEXITCODE -ne 0) { throw "$description failed with exit code $LASTEXITCODE" }
}

function Install-Pip([string[]]$Packages) {
    & $python -m pip install @Packages
    Assert-LastExit 'pip install'
}

function Enable-Msvc {
    if (-not $Vcvarsall -and (Get-Command cl.exe -ErrorAction SilentlyContinue) -and
            (Get-Command rc.exe -ErrorAction SilentlyContinue) -and $env:VCToolsInstallDir) { return }
    $candidate = $Vcvarsall
    if (-not $candidate) {
        $programFiles = ${env:ProgramFiles(x86)}
        if (-not $programFiles) { $programFiles = [Environment]::GetFolderPath('ProgramFilesX86') }
        $vswhere = if ($programFiles) { Join-Path $programFiles 'Microsoft Visual Studio\Installer\vswhere.exe' } else { $null }
        if ($vswhere -and (Test-Path $vswhere)) {
            $installations = & $vswhere -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            $candidate = $installations | ForEach-Object {
                Join-Path $_ 'VC\Auxiliary\Build\vcvarsall.bat'
            } | Where-Object { Test-Path $_ } | Select-Object -First 1
        }
    }
    if (-not $candidate -or -not (Test-Path $candidate)) {
        throw 'Install Visual Studio 2022 Desktop development with C++ and a Windows SDK, or pass -Vcvarsall <path>.'
    }
    $variables = & cmd.exe /d /s /c "call `"$candidate`" amd64 >nul && set"
    Assert-LastExit 'Visual Studio x64 environment initialization'
    foreach ($line in $variables) {
        $separator = $line.IndexOf('=')
        if ($separator -gt 0) {
            [Environment]::SetEnvironmentVariable($line.Substring(0, $separator), $line.Substring($separator + 1), 'Process')
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'Visual Studio initialization did not provide cl.exe' }
    if (-not (Get-Command rc.exe -ErrorAction SilentlyContinue)) {
        throw 'Windows SDK tools are missing; install the Windows SDK with the C++ desktop workload.'
    }
}

function Set-CudaEnvironment {
    if (-not (Test-Path (Join-Path $cuda 'bin\nvcc.exe'))) {
        throw 'CUDA Toolkit is missing from this Conda environment; run -Basic first.'
    }
    $env:CUDA_HOME = $cuda
    $env:CUDA_PATH = $cuda
    $env:PATH = "$env:CONDA_PREFIX;$env:CONDA_PREFIX\Scripts;$cuda\bin;$env:PATH"
    $env:LIB = "$cuda\lib;$env:LIB"
    $env:INCLUDE = "$cuda\include\targets\x64;$env:INCLUDE"
    $env:DISTUTILS_USE_SDK = '1'
}

function Get-BuildSource([string]$name, [string]$url, [string]$revision, [bool]$submodules) {
    $source = Join-Path $BuildRoot $name
    if (-not (Test-Path $source)) {
        & git -c core.autocrlf=false clone --no-checkout $url $source | Out-Null
        Assert-LastExit "Clone $name"
    }
    & git -C $source fetch $url $revision | Out-Null
    Assert-LastExit "Fetch $name revision"
    & git -C $source -c core.autocrlf=false checkout --detach $revision | Out-Null
    Assert-LastExit "Checkout $name"
    if ($submodules) {
        & git -C $source -c core.autocrlf=false submodule update --init --recursive | Out-Null
        Assert-LastExit "Submodules for $name"
    }
    return $source
}

function Apply-WindowsPatch([string]$source, [string]$relativePatch) {
    $patch = Join-Path $repo $relativePatch
    # An unpatched checkout makes the reverse check fail; this is expected.
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & git -C $source apply --reverse --check $patch 2>$null
        $alreadyApplied = $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($alreadyApplied) { return }
    & git -C $source apply $patch
    Assert-LastExit "Apply $relativePatch"
}

function Disable-LegacyCondaCompilerHooks {
    # CUDA's Windows Conda metapackage can install VS2017 compiler hooks even
    # when VS2022 is the selected compiler. Preserve those exact hooks for rollback.
    if ($env:VisualStudioVersion -notmatch '^17\.') { return }
    $metadata = Join-Path $env:CONDA_PREFIX 'conda-meta'
    if (-not (Get-ChildItem -LiteralPath $metadata -Filter 'vs2017_win-64-*.json' -ErrorAction SilentlyContinue)) { return }
    foreach ($name in @('vs2017_compiler_vars.bat', 'vs2017_get_vsinstall_dir.bat')) {
        $hookPath = Join-Path $env:CONDA_PREFIX "etc\conda\activate.d\$name"
        if (Test-Path -LiteralPath $hookPath) {
            $backupPath = "$hookPath.plsr-disabled"
            if (Test-Path -LiteralPath $backupPath) {
                throw "Both $hookPath and its backup exist; inspect them before rerunning setup."
            }
            Move-Item -LiteralPath $hookPath -Destination $backupPath
            Write-Output "Backed up incompatible VS2017 activation hook: $backupPath"
        }
    }
}

$native = $PyTorch3D -or $FlexGemm -or $DiffOctreeRast -or $NvDiffRast -or $CuMesh -or $OVoxel -or $NvDiffRec
# Check the compiler and SDK before creating an environment or downloading large packages.
if ($native -or $Basic) { Enable-Msvc }

if ($NewEnv) {
    $hook = & conda shell.powershell hook | Out-String
    Assert-LastExit 'Initialize Conda PowerShell'
    Invoke-Expression $hook
    & conda create -n plsr python=3.12 pip -y
    Assert-LastExit 'Create Conda environment plsr'
    & conda activate plsr
    Assert-LastExit 'Activate Conda environment plsr'
}
if (-not $env:CONDA_PREFIX) { throw 'Activate a Conda environment, or use -NewEnv.' }
$python = Join-Path $env:CONDA_PREFIX 'python.exe'
$cuda = Join-Path $env:CONDA_PREFIX 'Library'
if (-not (Test-Path $python)) { throw "Conda Python was not found: $python" }
$env:ATTN_BACKEND = 'sdpa'
$env:SPARSE_ATTN_BACKEND = 'sdpa'
$env:MAX_JOBS = [string]$MaxJobs

if ($NewEnv) {
    Install-Pip @("torch==$torchVersion", "torchvision==$torchvisionVersion", '--index-url', 'https://download.pytorch.org/whl/cu128')
}
if ($Basic) {
    if (-not (Test-Path (Join-Path $cuda 'bin\nvcc.exe'))) {
        & conda install --prefix $env:CONDA_PREFIX -c nvidia/label/cuda-12.8.0 cuda-toolkit=12.8 -y
        Assert-LastExit 'Install CUDA Toolkit 12.8'
    }
    Install-Pip @('setuptools', 'wheel', 'ninja', "triton-windows==$tritonVersion", '-r', (Join-Path $repo 'requirements.txt'))
}

if ($Basic) { Disable-LegacyCondaCompilerHooks }

if ($native) {
    if ($PyTorch3D -and (Join-Path $BuildRoot 'pytorch3d').Length -gt 80) {
        throw 'BuildRoot is too long for PyTorch3D; choose a short writable Windows path.'
    }
    Set-CudaEnvironment
    New-Item -ItemType Directory -Force $BuildRoot | Out-Null
}
if ($PyTorch3D) {
    $source = Get-BuildSource 'pytorch3d' 'https://github.com/facebookresearch/pytorch3d.git' '75ebeeaea0908c5527e7b1e305fbc7681382db47' $false
    Apply-WindowsPatch $source 'patches\pytorch3d-windows-cuda.patch'
    Install-Pip @('--no-build-isolation', $source)
}
if ($FlexGemm) {
    $source = Get-BuildSource 'flexgemm' 'https://github.com/JeffreyXiang/FlexGEMM.git' '1d83f0f6e457ddc8d948218ee79fb68808315688' $false
    Apply-WindowsPatch $source 'patches\flexgemm-windows.patch'
    Install-Pip @('--no-build-isolation', $source)
}
if ($DiffOctreeRast) {
    $source = Get-BuildSource 'diffoctreerast' 'https://github.com/JeffreyXiang/diffoctreerast.git' 'b09c20b84ec3aace4729e6e18a613112320eca3a' $true
    Install-Pip @('--no-build-isolation', $source)
}
if ($NvDiffRast) {
    $source = Get-BuildSource 'nvdiffrast' 'https://github.com/NVlabs/nvdiffrast.git' '253ac4fcea7de5f396371124af597e6cc957bfae' $false
    Install-Pip @('--no-build-isolation', $source)
}
if ($CuMesh) {
    $source = Get-BuildSource 'cumesh' 'https://github.com/JeffreyXiang/CuMesh.git' '12289e1062f0603f2f0d0771b02e1395d247f26f' $true
    Apply-WindowsPatch (Join-Path $source 'third_party\cubvh') 'patches\cubvh-windows-ascii.patch'
    Install-Pip @('--no-build-isolation', $source)
}
if ($OVoxel) {
    # Keep the selected CuMesh/FlexGEMM revisions instead of O-Voxel's Git defaults.
    Install-Pip @('--no-build-isolation', '--no-deps', (Join-Path $repo 'o-voxel'))
}
if ($MapAnything) {
    Install-Pip @("torchaudio==$torchVersion", '--index-url', 'https://download.pytorch.org/whl/cu128')
    Install-Pip @('imageio', 'uniception==0.1.6', 'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8', 'git+https://github.com/facebookresearch/map-anything.git@f065d13383ae84a7f80f0e0104a4d12afc7c7c82')
}
if ($NvDiffRec) {
    $source = Get-BuildSource 'nvdiffrec' 'https://github.com/JeffreyXiang/nvdiffrec.git' 'b296927cc7fd01c2ac1087c8065c4d7248f72da4' $false
    Install-Pip @('--no-build-isolation', $source)
}

# Save platform defaults in this environment and verify activation in a fresh process.
& conda env config vars set --prefix $env:CONDA_PREFIX ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa
Assert-LastExit 'Save Windows attention backends'
& conda run --no-capture-output --prefix $env:CONDA_PREFIX python --version
Assert-LastExit 'Fresh Conda activation'
