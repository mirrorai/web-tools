#!/usr/bin/env python

import sys
from subprocess import Popen

ip, user, password, configuration = sys.argv[1:]


# noinspection PyShadowingNames
def ssh(ip, cmd, ignore_return_code=False):
    full_cmd = "sshpass -p '{}' ssh -oStrictHostKeyChecking=no {}@{} {}".format(password, user, ip, cmd)
    print('ssh: {}'.format(full_cmd))
    call = Popen([full_cmd], shell=True)
    call.wait()
    if not ignore_return_code and call.returncode != 0:
        raise Exception('Get non-zero return code {} from command {}'.format(call.returncode, full_cmd))


# noinspection PyShadowingNames
def scp(ip, src, dst, ignore_return_code=False):
    full_cmd = "sshpass -p '{}' scp -r -oStrictHostKeyChecking=no {} {}@{}:{}".format(password, src, user, ip, dst)
    print('scp: {}'.format(full_cmd))
    call = Popen([full_cmd], shell=True)
    call.wait()
    if not ignore_return_code and call.returncode != 0:
        raise Exception('Get non-zero return code {} from command {}'.format(call.returncode, full_cmd))


ssh(ip, 'sudo killall python', ignore_return_code=True)
ssh(ip, 'sudo killall uwsgi_python', ignore_return_code=True)
ssh(ip, 'sudo killall celery', ignore_return_code=True)

ssh(ip, 'sudo rm -rf /home/local/queue')
ssh(ip, 'mkdir -p /home/local/queue/queue')
for item in [
        'queue', 'wsgi.py', 'run_wsgi.sh', 'config.py', 'manage.py', 'requirements.txt', 'migrations', '.bowerrc',
        'bower.json', 'package.json', 'db_population_staging.json', 'db_population_production.json', 'static',
        'templates'
        ]:
    scp(ip, item, '/home/local/queue')

scp(ip, 'ci/bundle.sh', '/home/local/bundle.sh')
ssh(ip, 'chmod +x /home/local/bundle.sh')

scp(ip, 'ci/rc.local', '/home/local/rc.local')
ssh(ip, 'sudo mv /home/local/rc.local /etc/rc.local')
ssh(ip, 'sudo chmod +x /etc/rc.local')

ssh(ip, "\'grep -q -F \"QUEUE\" {file} || (echo \"QUEUE_CONFIG={conf}\" | sudo tee -a {file})\'".format(
        conf=configuration,
        file='/etc/environment'))

ssh(ip, 'sudo reboot', ignore_return_code=True)
