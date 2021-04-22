


zkl@zkl-tower:~$ supervisord -c /etc/supervisor/supervisord.conf


supervisorctl status

ps -ef | grep supervisor
kill xxx

supervisorctl kill jianpan_4_9

supervisorctl restart jianpan_4_9

supervisorctl reload


tail -f 100 jp.supervisor.stdout.4.9.log



curl -H "Content-Type: application/json" -X POST -d '{'modelId':1,'D04:FWF':1608.16601,'D04:SELPRESS':17.173832}' http://0.0.0.0:8383/predict