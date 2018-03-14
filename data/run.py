import time
import os
import subprocess


start_time_str = time.strftime('%y.%m.%d-%H:%M:%S', time.localtime(time.time()))
start_time = time.time()
error_ratios = ([0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4])
dir_name = 'data'

for error_ratio in error_ratios:
    low = error_ratio[0]
    high = error_ratio[1]
    print('\n=====================================')
    print('Creating dataset {} ~ {} ...'.format(low, high))
    print('=====================================\n')
    subprocess.call(['python3 data_process.py {} {} {}'.format(low, high, dir_name)], shell=True)
    archive_name = 'data-{}~{}.7z'.format(low, high)
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
