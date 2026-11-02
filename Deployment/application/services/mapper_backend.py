#!/usr/bin/env python3
"""Inspect the running mapping backend without changing services."""
import argparse
import json
from pathlib import Path
import subprocess


PIPELINE_UNITS = ('dapeng-wavemap.service', 'dapeng-nvblox.service', 'algorithm-pipeline.service')
BUSY_STATES = ('active', 'activating', 'reloading', 'deactivating')


def mapper_from_command(arguments, root):
    launch = Path(root).resolve() / 'Deployment/application'
    if not any(Path(value) in (launch/'services/run_pipeline.py', launch/'scripts/start_pipeline.sh')
               for value in arguments):
        raise RuntimeError('算法服务的运行命令不属于当前部署目录，拒绝猜测建图后端。')
    mapper = 'wavemap'
    for index, value in enumerate(arguments):
        if value == '--mapper':
            if index+1 == len(arguments):
                raise RuntimeError('算法启动命令缺少 --mapper 参数值。')
            mapper = arguments[index+1]
        elif value.startswith('--mapper='):
            mapper = value.split('=',1)[1]
    if mapper not in ('wavemap', 'nvblox'):
        raise RuntimeError(f'无法识别运行中的建图后端：{mapper}')
    return mapper


def inspect_backend(root, read_state=None, read_command=None):
    if read_state is None:
        def read_state(unit):
            result = subprocess.run(['systemctl','--user','show',unit,
                                     '-p','ActiveState','-p','MainPID'],
                                    text=True,capture_output=True)
            fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
            if 'ActiveState' not in fields:
                raise RuntimeError(f'无法读取 {unit} 状态：{result.stderr.strip()}')
            return fields
    if read_command is None:
        def read_command(pid):
            return Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
    running = [(unit,read_state(unit)) for unit in PIPELINE_UNITS]
    running = [(unit,state) for unit,state in running if state.get('ActiveState') in BUSY_STATES]
    if len(running)>1:
        raise RuntimeError('多个算法服务同时运行：' + ', '.join(unit for unit,_ in running))
    if not running:
        return {'unit':'','mapper':'','state':'inactive','pid':0,'navigation':False}
    unit,state = running[0]
    if state['ActiveState'] == 'deactivating':
        raise RuntimeError(f'{unit} 正在停止，请等待结束。')
    pid = int(state.get('MainPID','0'))
    if not pid:
        raise RuntimeError(f'{unit} 尚未提供运行进程，请稍后重试。')
    arguments=read_command(pid)
    return {'unit':unit,'mapper':mapper_from_command(arguments,root),
            'state':state['ActiveState'],'pid':pid,'navigation':'--navigation' in arguments}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--format',choices=('json','shell'),default='json')
    args=parser.parse_args()
    try:
        info=inspect_backend(args.root)
    except (OSError,ValueError,RuntimeError) as error:
        parser.exit(1,str(error)+'\n')
    if args.format=='shell':
        print(info['unit'] or '-',info['mapper'] or '-',info['state'])
    else:
        print(json.dumps(info))


if __name__=='__main__':
    main()
