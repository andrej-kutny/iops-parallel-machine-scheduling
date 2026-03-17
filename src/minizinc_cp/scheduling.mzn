% Parallel machine scheduling: minimize makespan with release dates and
% sequence-dependent setup times
%
% Sets / indices:
%   j in 1..n    jobs
%   k in 1..m    machines
%
% Decision variables:
%   assign[j]    machine chosen for job j
%   start[j]     start time of job j on its assigned machine
%   end[j]       completion time of job j on its assigned machine

int: n;
int: m;
int: horizon;

% capable[j,k] = 1  if job j can run on machine k, 0 otherwise
array[1..n, 1..m] of 0..1: capable;

% duration[j,k] = processing time of job j on machine k
array[1..n, 1..m] of int: duration;

% release[j,k] = release time of job j on machine k
array[1..n, 1..m] of int: release;

% setup[i,j,k] = setup time on machine k when j is processed immediately after i
array[1..n, 1..n, 1..m] of int: setup;

array[1..n] of var 1..m: assign;
array[1..n] of var 0..horizon: start;
array[1..n] of var 0..horizon: end;

% Each job must be assigned to a capable machine
constraint forall(j in 1..n)(capable[j, assign[j]] = 1);

% Start time must respect the job's release date on the chosen machine
constraint forall(j in 1..n)(start[j] >= release[j, assign[j]]);

% Completion time = start + processing time on the chosen machine
constraint forall(j in 1..n)(end[j] = start[j] + duration[j, assign[j]]);

% No-overlap constraint on each machine:
% For any two distinct jobs i and j on the same machine, either:
%   - i finishes (plus setup) before j starts, or
%   - j finishes (plus setup) before i starts
% If they are on different machines, no ordering constraint is needed
constraint forall(i, j in 1..n where i < j)(
  (assign[i] != assign[j]) \/
  (end[i] + setup[i, j, assign[i]] <= start[j]) \/
  (end[j] + setup[j, i, assign[j]] <= start[i])
);

% Objective: minimize the makespan (latest completion time over all jobs)
% Search: assign machine per job first (first_fail), then start times
solve :: int_search(assign, first_fail, indomain_min)
    minimize max(j in 1..n)(end[j]);
