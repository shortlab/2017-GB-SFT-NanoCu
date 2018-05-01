

inputfile=/home/jinmiao/projects/lammps-16Feb16/myproj/GB_Cu/input.minimize
start=11000
step=1000
end=99000
for num in $(seq ${start} ${step} ${end}); do
    dumpfile=dump_relaxsteps_${num}.data
    outfile=dump.data
    echo ${dumpfile}
    if [ ! -f "${dumpfile}" ];then
       continue
    fi
    python /home/jinmiao/projects/lammps-16Feb16/mytools/dump2read_data/dump2input.py ${dumpfile} ${outfile}
    mpirun /home/jinmiao/projects/lammps-16Feb16/src/lmp_mpi -screen none < ${inputfile}  
    if [ ! -f "dump_min.data" ];then
       continue
    fi
    mv dump_min.data dump_relaxsteps_${num}_min.data
    rm ${outfile}
    rm ${dumpfile}
done
