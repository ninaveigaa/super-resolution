#!/bin/bash
set -e

echo "== 1. Criando estrutura de pastas =="
mkdir -p src/dashboard
mkdir -p models
mkdir -p pipelines
mkdir -p scripts
mkdir -p jobs
mkdir -p data outputs logs
mkdir -p dualedsr_tensorflow
mkdir -p archive

echo "== 2. Movendo arquivos existentes =="
git mv models/hat.py archive/hat_draft.py 2>/dev/null || echo "  (pulado: models/hat.py não encontrado ou já movido)"

git mv scripts/train.py pipelines/train_3d.py 2>/dev/null || echo "  (pulado: scripts/train.py não encontrado)"
git mv scripts/image_comparison.py archive/image_comparison_old.py 2>/dev/null || echo "  (pulado: image_comparison.py não encontrado)"
rmdir scripts 2>/dev/null || true
mkdir -p scripts

touch src/dashboard/__init__.py
touch src/dashboard/generate_dashboard.py
touch src/dashboard/template.html

echo "== 3. Criando arquivos novos (vazios, a preencher) =="
touch models/base.py
touch pipelines/infer_3d.py pipelines/train_2d.py pipelines/infer_2d.py
touch scripts/download_data.sh scripts/sync_from_drive.sh scripts/sync_to_drive.sh
chmod +x scripts/*.sh
touch jobs/train_dualedsr.sh jobs/train_dualhat.sh jobs/train_2d_study.sh
chmod +x jobs/*.sh
touch data/.gitkeep outputs/.gitkeep logs/.gitkeep

echo "== 4. Limpeza (venv, checkpoints, cache, lixo do macOS) =="
git rm -r --cached super-resolution/ 2>/dev/null || true
git rm -r --cached outputs/ 2>/dev/null || true
git rm -r --cached test_checkpoints/ 2>/dev/null || true
git rm --cached .DS_Store 2>/dev/null || true
find . -name "__pycache__" -not -path "./super-resolution/*" -exec git rm -r --cached {} \; 2>/dev/null || true

echo "== 5. Pronto. Revise 'git status' antes de commitar. =="
git status
