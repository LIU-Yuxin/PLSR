#!/usr/bin/env bash
# Linux installation, following TRELLIS.2's selectable dependency groups.

plsr_setup() {
    local plsr_root
    plsr_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return
    local torch_version=2.9.1 torchvision_version=0.24.1 flash_attn_version=2.8.3
    local option new_env=false basic=false flash_attn=false pytorch3d=false
    local flexgemm=false diffoctreerast=false nvdiffrast=false cumesh=false
    local ovoxel=false mapanything=false nvdiffrec=false

    if [[ $# -eq 0 ]]; then set -- --help; fi
    for option in "$@"; do
        case "$option" in
            -h|--help)
                cat <<'EOF'
Usage: . ./setup.sh [OPTIONS]

  --new-env        Create and activate Conda environment 'plsr' (Python 3.12),
                   then install PyTorch/torchvision for CUDA 12.8.
  --basic          Install inference Python dependencies and build tools.
  --flash-attn     Install FlashAttention (optional when using SDPA).
  --pytorch3d      Install PyTorch3D for local attention.
  --flexgemm       Install FlexGEMM for sparse convolution and grid sampling.
  --diffoctreerast Install the patch visibility renderer.
  --nvdiffrast     Install the texture baking renderer.
  --cumesh         Install CuMesh for mesh processing/export.
  --o-voxel        Install the bundled O-Voxel source.
  --mapanything    Install camera estimation and utils3d dependencies.
  --nvdiffrec      Install the camera anchor renderer.
  -h, --help       Show this help.

Options can be combined or run separately. Prepare a CUDA Toolkit and C++
compiler before building extensions. See docs/INSTALL.md for the full command.
With an existing environment, omit --new-env and use 'bash setup.sh ...'.
EOF
                return 0
                ;;
            --new-env) new_env=true ;;
            --basic|--python-deps) basic=true ;;
            --flash-attn) flash_attn=true ;;
            --pytorch3d) pytorch3d=true ;;
            --flexgemm) flexgemm=true ;;
            --diffoctreerast) diffoctreerast=true ;;
            --nvdiffrast) nvdiffrast=true ;;
            --cumesh) cumesh=true ;;
            --o-voxel) ovoxel=true ;;
            --mapanything) mapanything=true ;;
            --nvdiffrec) nvdiffrec=true ;;
            *) echo "Unknown option: $option. See 'bash setup.sh --help'." >&2; return 2 ;;
        esac
    done

    if [[ "$(uname -s)" != Linux ]]; then
        echo 'This installer targets Linux/WSL; see docs/INSTALL_WINDOWS.md for Windows.' >&2
        return 2
    fi

    if $new_env; then
        local conda_base
        conda_base="$(conda info --base)" || return
        source "$conda_base/etc/profile.d/conda.sh" || return
        conda create -n plsr python=3.12 pip -y || return
        conda activate plsr || return
        python -m pip install "torch==$torch_version" "torchvision==$torchvision_version" \
            --index-url https://download.pytorch.org/whl/cu128 || return
    fi

    if $basic; then
        python -m pip install setuptools wheel ninja \
            -r "$plsr_root/requirements.txt" || return
    fi
    if $flash_attn; then
        python -m pip install "flash-attn==$flash_attn_version" --no-build-isolation || return
    fi
    if $pytorch3d; then
        python -m pip install --no-build-isolation \
            'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47' || return
    fi
    if $flexgemm; then
        python -m pip install --no-build-isolation \
            'git+https://github.com/JeffreyXiang/FlexGEMM.git@1d83f0f6e457ddc8d948218ee79fb68808315688' || return
    fi
    if $diffoctreerast; then
        python -m pip install --no-build-isolation \
            'git+https://github.com/JeffreyXiang/diffoctreerast.git@b09c20b84ec3aace4729e6e18a613112320eca3a' || return
    fi
    if $nvdiffrast; then
        python -m pip install --no-build-isolation \
            'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae' || return
    fi
    if $cumesh; then
        python -m pip install --no-build-isolation \
            'git+https://github.com/JeffreyXiang/CuMesh.git@12289e1062f0603f2f0d0771b02e1395d247f26f' || return
    fi
    if $ovoxel; then
        # Keep the selected CuMesh/FlexGEMM revisions instead of O-Voxel's Git defaults.
        python -m pip install --no-build-isolation --no-deps "$plsr_root/o-voxel" || return
    fi
    if $mapanything; then
        python -m pip install "torchaudio==$torch_version" \
            --index-url https://download.pytorch.org/whl/cu128 || return
        python -m pip install imageio "uniception==0.1.6" \
            'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8' \
            'git+https://github.com/facebookresearch/map-anything.git@f065d13383ae84a7f80f0e0104a4d12afc7c7c82' || return
    fi
    if $nvdiffrec; then
        python -m pip install --no-build-isolation \
            'git+https://github.com/JeffreyXiang/nvdiffrec.git@b296927cc7fd01c2ac1087c8065c4d7248f72da4' || return
    fi
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    plsr_setup "$@"
    exit "$?"
else
    plsr_setup "$@"
    return "$?"
fi
