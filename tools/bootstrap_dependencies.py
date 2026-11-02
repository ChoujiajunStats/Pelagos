#!/usr/bin/env python3
"""Inspect dependency pins; --apply clones only missing Git source directories.

This does not install packages, retrieve LFS weights, restore manual snapshots,
build images, or launch the simulation. Existing paths are never altered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def contained_path(root: Path, relative: str, *, dependency: bool = False) -> Path:
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts or not part.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    if dependency and (part.parts[0] != 'TP' or len(part.parts) < 3):
        raise ValueError(f'Dependency must be below TP/<category>/<name>: {relative}')
    resolved = (root / part).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes project root: {relative}')
    return root / part


def load_manifest(path: Path, root: Path) -> list[dict]:
    document = json.loads(path.read_text())
    if document.get('schema_version') != 1:
        raise ValueError('Unsupported dependency manifest schema')
    entries = document['dependencies']
    identities, targets = set(), set()
    for entry in entries:
        identity, relative = entry['id'], entry['path']
        if identity in identities or relative in targets:
            raise ValueError(f'Duplicate dependency: {identity} / {relative}')
        identities.add(identity); targets.add(relative)
        contained_path(root, relative, dependency=True)
        if entry.get('restore', {}).get('automatic'):
            if not re.fullmatch(r'[0-9a-f]{40}', entry.get('commit') or ''):
                raise ValueError(f'{identity}: a full fixed Git commit is required')
            if not isinstance(entry.get('url'), str) or not entry['url']:
                raise ValueError(f'{identity}: Git URL is required')
        for patch in entry.get('restore', {}).get('patches', []):
            file = contained_path(root, patch['path'])
            if hashlib.sha256(file.read_bytes()).hexdigest() != patch['sha256']:
                raise ValueError(f'{identity}: patch hash mismatch: {patch["path"]}')
    for entry in entries:
        if entry.get('managed_by') and entry['managed_by'] not in identities:
            raise ValueError(f'{entry["id"]}: unknown parent dependency')
    return entries


def git(*arguments: str, env: dict) -> str:
    result = subprocess.run(['git', *arguments], env=env, check=True,
                            stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()


def bootstrap(entries: list[dict], root: Path, *, apply: bool = False,
              selected: set[str] | None = None) -> tuple[list[dict], bool]:
    root = root.resolve()
    records, created = [], set()
    incomplete = False
    # Avoid automatic LFS downloads, including inside recursive submodules.
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')
    for entry in entries:
        identity = entry['id']
        if selected and identity not in selected:
            continue
        target = contained_path(root, entry['path'], dependency=True)
        record = dict(id=identity, path=entry['path'])
        if entry.get('managed_by') in created:
            # Parent checkout/submodule update created this path during this call.
            # Verify its recorded pin without changing the submodule ourselves.
            actual = git('-C', str(target), 'rev-parse', 'HEAD', env=env)
            if actual != entry['commit']:
                raise ValueError(f'{identity}: parent restored unexpected submodule commit {actual}')
            record['result'] = 'RESTORED_BY_PARENT'
        elif target.exists() or target.is_symlink():
            record['result'] = 'EXISTS_UNCHANGED'
        elif not entry.get('restore', {}).get('automatic'):
            record['result'] = 'MANUAL_REQUIRED'
            record['reason'] = entry.get('restore', {}).get('reason', 'No verified automatic source restoration')
            incomplete = True
        elif not apply:
            record.update(result='WOULD_CLONE_PIN', commit=entry['commit'], url=entry['url'])
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Git refuses an occupied destination. Do not reset/pull/clean an
            # existing checkout or delete a partial clone after an error.
            git('clone', '--no-checkout', '--', entry['url'], str(target), env=env)
            git('-C', str(target), 'checkout', '--detach', entry['commit'], env=env)
            actual = git('-C', str(target), 'rev-parse', 'HEAD', env=env)
            if actual != entry['commit']:
                raise ValueError(f'{identity}: checkout did not produce exact pin')
            git('-C', str(target), 'submodule', 'update', '--init', '--recursive', env=env)
            for patch in entry.get('restore', {}).get('patches', []):
                file = contained_path(root, patch['path'])
                git('-C', str(target), 'apply', '--check', str(file), env=env)
                git('-C', str(target), 'apply', str(file), env=env)
            record.update(result='CLONED_PIN', commit=actual)
            created.add(identity)
        records.append(record)
    return records, incomplete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'TP/dependencies.lock.json')
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='Destination project root (default: this checkout)')
    parser.add_argument('--apply', action='store_true', help='Clone missing pinned sources; uses network')
    parser.add_argument('--only', action='append', help='Limit to dependency id; repeat as needed')
    args = parser.parse_args()
    try:
        args.root = args.root.resolve()
        entries = load_manifest(args.manifest, args.root)
        selected = set(args.only or [])
        unknown = selected - {entry['id'] for entry in entries}
        if unknown:
            raise ValueError('Unknown dependency ids: '+', '.join(sorted(unknown)))
        records, incomplete = bootstrap(entries, args.root, apply=args.apply, selected=selected)
        print(json.dumps(dict(mode='apply' if args.apply else 'dry_run',
                              lfs='skipped; model/data files are separate requirements',
                              existing_paths='never modified or certified by this script',
                              dependencies=records), indent=2))
        return 2 if args.apply and incomplete else 0
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f'Dependency restoration failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
