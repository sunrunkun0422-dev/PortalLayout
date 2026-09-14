#!/usr/bin/env python3
"""Local detached cache -> joint-training supervisor; no network dependency.

Launch with subprocess.Popen(start_new_session=True), redirecting to a real log
file. Epoch checkpoints and completed cache rows survive process restarts.
"""
import argparse,datetime,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path

def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.config.read_text());root=Path(config['repo']);job=Path(config['job_dir']);job.mkdir(parents=True,exist_ok=True)
    lock=(job/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (job/'supervisor.pid').write_text(str(os.getpid())+'\n');child=None;stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/observable_mpl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    state={'pid':os.getpid(),'status':'starting','network_required':False,'job_config':str(a.config.resolve())}
    def save(**kw):
        state.update(kw);state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(job/'job_status.json',state)
    if (job/'COMPLETED.json').exists():print('Job already complete.',flush=True);return 0
    for stage in ['cache','train']:
        for attempt in range(1,config.get('maximum_attempts',3)+1):
            command=list(config[stage+'_command'])
            if stage=='train' and (Path(config['training_output'])/'last.pt').exists():command+=['--resume','auto']
            if stopping:save(status='stopped');return 130
            print('START',stage,'attempt',attempt,flush=True)
            save(status='running',stage=stage,attempt=attempt,command=command)
            with (job/(stage+'.log')).open('a',buffering=1) as log:
                child=subprocess.Popen(command,cwd=str(root),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                save(child_pid=child.pid)
                while child.poll() is None:
                    time.sleep(10);save(heartbeat=True)
                result=child.returncode
            save(last_exit_code=result)
            if stopping:save(status='stopped');return 130
            if result==0:break
            if attempt==config.get('maximum_attempts',3):save(status='failed');return result
            print('Retrying from completed cache rows / last epoch checkpoint in 30 seconds.',flush=True);time.sleep(30)
    output=Path(config['training_output']);metrics=json.loads((output/'best_metrics.json').read_text())
    completed={'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'best_checkpoint':str(output/'best.pt'),'metrics':metrics,
        'evaluation':'internal official-train house monitor, not official val/test','human_verified':False}
    atomic(job/'COMPLETED.json',completed);save(status='completed',stage='done')
    (job/'训练完成报告.md').write_text('# ObservablePairs 端到端探索性训练完成\n\n最佳权重：`'+str(output/'best.pt')+'`。\n\n结果来自官方训练房屋内的独立监控划分；预训练初始化可能见过这些房屋，不作为未见房屋泛化结论。没有使用正式 val/test；标签仍为自动提议。\n\n```json\n'+json.dumps(metrics,ensure_ascii=False,indent=2)+'\n```\n')
    print('TRAINING COMPLETE',output/'best.pt',flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
