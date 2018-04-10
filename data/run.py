import time
import os
import subprocess


start_time_str = time.strftime('%y.%m.%d-%H:%M:%S', time.localtime(time.time()))
start_time = time.time()
error_ratios = [0]
dir_name = 'data'

for error_ratio in error_ratios:
    print('\n=====================================')
    print('Creating dataset with error ratio: {} ...'.format(error_ratio))
    print('=====================================\n')
    subprocess.call(['python3 data_process.py {} {}'.format(error_ratio, dir_name)], shell=True)
    archive_name = 'data-{}.7z'.format(error_ratio)
    subprocess.call(['7z a {} {}'.format(archive_name, dir_name)], shell=True)
    subprocess.call(['rm -r {}'.format(dir_name)], shell=True)


end_time_str = time.strftime('%y.%m.%d %H:%M:%S', time.localtime(time.time()))
end_time = time.time()

run_seconds = end_time - start_time
run_hours = int(run_seconds // 3600)
run_seconds = run_seconds - (run_hours * 3600)
run_mins = int(run_seconds // 60)
run_seconds = run_seconds - (run_mins * 60)
print('\n================================')
print('Start time:\t{}'.format(start_time_str))
print('End time:\t{}'.format(end_time_str))
print('Total time:\t{}h {}m {:.2f}s'.format(run_hours, run_mins, run_seconds))
