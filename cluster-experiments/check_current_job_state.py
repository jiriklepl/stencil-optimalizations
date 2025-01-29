import os
import re

RUNNING_JOBS_OUTDIR = 'experiments-outputs'
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def get_fnames_in_dir(dir):
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

def get_squeue_jobs():
    return os.popen('squeue').read()

def get_job_run_time_secs(fname: str, squeue_out: str) -> int:
    split_parts = re.split(r'[^a-zA-Z0-9]+', fname)

    for part in split_parts:
        try:
            id = int(part)
            continue
        except:
            pass
    
    jobs = [l.strip() for l in squeue_out.split('\n')]

    for job in jobs:
        if job.startswith(str(id)):
            job_line = job
    
    job_parts = [p for p in job_line.split() if p]

    time_parts = job_parts[5].split(':')
    time_parts = time_parts if len(time_parts) == 3 else ['0', *time_parts]

    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = int(time_parts[2])

    total_secs = hours * 3600 + minutes * 60 + seconds

    return total_secs

def print_state_of_job(fname):
    with open(fname, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    running_script_name = [l for l in lines if l.startswith('SCRIPT:')][0].split()[1]
    experiments_completed = len([l for l in lines if l.startswith('exp-')])

    with open(running_script_name, 'r') as f:
        running_script_content = f.read()

    script_lines = running_script_content.split('\n')
    total_experiments = len([l for l in script_lines if l.startswith('echo "exp-')])

    percent_done = experiments_completed / total_experiments * 100

    squeue_out = get_squeue_jobs()
    try:
        current_run_time = get_job_run_time_secs(fname, squeue_out)
    except:
        print (BLUE + 'Job:' + RESET, YELLOW + fname + RESET, BLUE + 'is not running' + RESET + '\n')
        return

    estimated_total_time = current_run_time / percent_done * 100
    est_time_to_finish = estimated_total_time - current_run_time

    est_str = f'{est_time_to_finish // 3600:3.0f} h {est_time_to_finish % 3600 // 60:2.0f} m {est_time_to_finish % 60:.0f} s'

    print(BLUE + 'Job:' + RESET,
          YELLOW + fname + RESET,
          BLUE + 'completed:' + RESET,
          YELLOW + str(experiments_completed) + RESET,
          BLUE + 'out of' + RESET,
          YELLOW + str(total_experiments) + RESET,
          BLUE + 'experiments\n' 
          + '   i.e.' + RESET,
          YELLOW + f'{percent_done:.2f} %' + RESET,
          BLUE + ' est. time to finish:' + RESET,
          YELLOW + est_str + RESET)
    
    print()

print()    
for fname in get_fnames_in_dir(RUNNING_JOBS_OUTDIR):
    print_state_of_job(os.path.join(RUNNING_JOBS_OUTDIR, fname))
print()
