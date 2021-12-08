cd /sys/fs/cgroup/cpuset/
mkdir cluster
mkdir partition
echo 0 > cpuset.sched_load_balance
cd cluster/
echo 1-7 > cpuset.cpus
echo 0 > cpuset.mems
echo 1 > cpuset.cpu_exclusive 
ps -eLo lwp | while read thread; do echo $thread > tasks ; done
cd ../partition/
echo 1 > cpuset.cpu_exclusive 
echo 0 > cpuset.mems 
echo 0 > cpuset.cpus
echo $$ > tasks 