# -*- coding: utf-8 -*-
"""
Git 기반 배포 스크립트
- 코드: git pull로 동기화 (가볍고 빠름)
- 모델: 필요시에만 scp로 전송
"""
import paramiko
import subprocess
import os

# 연결 정보
HOSTNAME = 'web01.digitalag.kr'
USERNAME = 'hj5258'
PASSWORD = 'wjsgPwls'
REMOTE_DIR = '/home/hj5258/P1_apple_RGB'
REPO_URL = 'https://github.com/202116302/P1_apple_RGB.git'

def run_local(cmd):
    """로컬 명령 실행"""
    print(f'[LOCAL] {cmd}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='C:/code/P1_apple_RGB')
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f'Error: {result.stderr}')
    return result.returncode == 0

def run_remote(ssh, cmd):
    """원격 명령 실행"""
    print(f'[REMOTE] {cmd}')
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=300)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out:
        print(out)
    if err and 'warning' not in err.lower():
        print(f'STDERR: {err}')
    return stdout.channel.recv_exit_status() == 0

def git_push_local():
    """로컬 변경사항 push"""
    print('\n' + '='*50)
    print('1. 로컬 Git 상태 확인')
    print('='*50)

    # 상태 확인
    subprocess.run('git status --short', shell=True, cwd='C:/code/P1_apple_RGB')

    response = input('\n변경사항을 커밋하고 push 하시겠습니까? (y/n): ').strip().lower()
    if response == 'y':
        msg = input('커밋 메시지: ').strip() or 'Update'
        run_local('git add -A')
        run_local(f'git commit -m "{msg}"')
        run_local('git push origin master')
    else:
        print('Push 건너뜀')

def deploy():
    """서버에 배포"""
    print('\n' + '='*50)
    print('2. 서버 배포 시작')
    print('='*50)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f'Connecting to {HOSTNAME}...')
        ssh.connect(HOSTNAME, username=USERNAME, password=PASSWORD)
        print('Connected!')

        # 저장소 존재 여부 확인
        stdin, stdout, stderr = ssh.exec_command(f'test -d {REMOTE_DIR}/.git && echo "exists"')
        repo_exists = 'exists' in stdout.read().decode()

        if repo_exists:
            # git pull
            print('\n기존 저장소 업데이트 (git pull)...')
            run_remote(ssh, f'cd {REMOTE_DIR} && git fetch origin && git reset --hard origin/master')
        else:
            # git clone
            print('\n저장소 클론 중...')
            run_remote(ssh, f'rm -rf {REMOTE_DIR}')
            run_remote(ssh, f'git clone {REPO_URL} {REMOTE_DIR}')

        # Docker 빌드 및 실행
        print('\n' + '='*50)
        print('3. Docker 빌드 및 실행')
        print('='*50)

        commands = [
            f'cd {REMOTE_DIR} && docker build -t p1-apple-rgb .',
            'docker stop p1-apple-rgb 2>/dev/null; docker rm p1-apple-rgb 2>/dev/null; echo "Cleaned up"',
            'docker run -d --name p1-apple-rgb -p 6000:8000 p1-apple-rgb',
            'docker ps | grep p1-apple-rgb'
        ]

        for cmd in commands:
            run_remote(ssh, cmd)

        print('\n' + '='*50)
        print('배포 완료!')
        print('URL: http://web01.digitalag.kr:6000')
        print('='*50)

    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        ssh.close()

def upload_models():
    """모델 파일만 별도 업로드 (필요시)"""
    print('\n' + '='*50)
    print('모델 파일 업로드')
    print('='*50)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(HOSTNAME, username=USERNAME, password=PASSWORD)
        sftp = ssh.open_sftp()

        local_models = 'C:/code/P1_apple_RGB/models'
        remote_models = f'{REMOTE_DIR}/models'

        # 원격 디렉토리 생성
        run_remote(ssh, f'mkdir -p {remote_models}')

        # 모델 파일 업로드
        for filename in os.listdir(local_models):
            if filename.endswith('.pth'):
                local_path = os.path.join(local_models, filename)
                remote_path = f'{remote_models}/{filename}'
                size_mb = os.path.getsize(local_path) / 1024 / 1024
                print(f'Uploading {filename} ({size_mb:.1f} MB)...')
                sftp.put(local_path, remote_path)
                print(f'  Done!')

        sftp.close()
        print('모델 업로드 완료!')

    except Exception as e:
        print(f'Error: {e}')
    finally:
        ssh.close()

if __name__ == '__main__':
    print('='*50)
    print('Git 기반 배포 스크립트')
    print('='*50)
    print('\n옵션:')
    print('1. 전체 배포 (git push + pull + docker)')
    print('2. 서버만 배포 (git pull + docker)')
    print('3. 모델 파일만 업로드')

    choice = input('\n선택 (1/2/3): ').strip()

    if choice == '1':
        git_push_local()
        deploy()
    elif choice == '2':
        deploy()
    elif choice == '3':
        upload_models()
    else:
        print('잘못된 선택')
