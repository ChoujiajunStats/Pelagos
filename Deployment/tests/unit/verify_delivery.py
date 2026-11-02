#!/usr/bin/env python3
"""Check delivery layout and contracts without connecting to robot hardware."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT = ROOT / 'Deployment'


def verify():
    architecture = json.loads((DEPLOYMENT / 'config/architecture.json').read_text())
    entrypoint = ROOT / architecture['entrypoint']
    assert entrypoint.is_file() and entrypoint.stat().st_mode & 0o111, entrypoint
    contexts = architecture['contexts']
    visited, active = set(), set()

    def visit(name):
        if name in active:
            raise ValueError('Cyclic context dependency: ' + name)
        if name in visited:
            return
        active.add(name)
        assert (ROOT / contexts[name]['path']).is_dir(), name
        for dependency in contexts[name]['dependencies']:
            visit(dependency)
        active.remove(name)
        visited.add(name)

    for name in contexts:
        visit(name)
    counts = {'python_syntax': 0, 'shell_syntax': 0, 'pure_domain_modules': 0,
              'context_dependencies': len(contexts), 'binary_packages': 0}
    for path in DEPLOYMENT.rglob('*.py'):
        if '__pycache__' not in path.parts:
            ast.parse(path.read_text(), filename=str(path))
            counts['python_syntax'] += 1
    for path in DEPLOYMENT.rglob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True)
        counts['shell_syntax'] += 1
    forbidden = set(architecture['pure_domain_forbidden_imports'])
    for relative in architecture['pure_domain_files']:
        tree = ast.parse((ROOT / relative).read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
        assert not imports & forbidden, (relative, imports & forbidden)
        counts['pure_domain_modules'] += 1
    required = [
        'TP/Mapping/nvblox/nvblox_ros/nvblox_core/CMakeLists.txt',
        'TP/Mapping/wavemap/library/cpp/CMakeLists.txt',
        'TP/Localization/okvis2/CMakeLists.txt',
        'TP/Stereo/LiteAnyStereoV2/core/models.py',
        'TP/Planner/wavestar/src/multi_res_theta_star.cc',
        'TP/Controller/disaware_learning/control/jepa_mppi.py',
        'Deployment/config/templates/okvis2-template.yaml',
    ]
    for relative in required:
        assert (ROOT / relative).is_file() and (ROOT / relative).stat().st_size, relative
    configuration = json.loads((DEPLOYMENT / 'contexts/control/config/config.json').read_text())
    assert configuration['motor_channels'] == ['R1', 'L1', 'R3', 'L3', 'R2', 'L2']
    assert configuration['motor_signs'] == [1, 1, -1, -1, -1, 1]
    assert configuration['hardware_response_validated'] is False
    checkpoint = ROOT / 'TP/Controller/disaware_learning/checkpoints/stage2_zhat_response_potential_full4096_four_modal_e5.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == 'e4e09e92f5ae00a6183c6f698ad32d9bcf671705d93d6b81e44e14d43f67974d'
    infrastructure = DEPLOYMENT / 'infrastructure/packages/nvblox'
    packages = json.loads((infrastructure / 'binary-packages.lock.json').read_text())
    for package in packages['packages']:
        data = (infrastructure / package['local_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == package['SHA256'], package['Package']
        counts['binary_packages'] += 1
    for path in ROOT.rglob('*'):
        if path.is_symlink():
            assert path.resolve().is_relative_to(ROOT), 'External symlink: ' + str(path)
    return {'status': 'passed', 'checks': counts,
            'robot_contacted': False, 'fresh_jetson_install_validated': False,
            'ros2_humble_cuda_runtime_validated_by_this_check': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify()
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
