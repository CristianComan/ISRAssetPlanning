% First JISR Asset Planning Model


enum CR;
%enum ASSET;
enum INTELL;
enum NAI;
enum BASE;

% need a better to define ASSET0 in a data file - cannot retrieve the NO-ASSET element
enum ASSET = {RQ_1A_Predator, MQ_9_Reaper, RQ_4_Global_Hawk, RC_135_Rivet_Joint, E_8_JSTARS, U_2_Dragon_Lady, F_16, NO_ASSET};

% CRL
array[CR] of INTELL: required_int;
array[CR] of NAI: location;
array[CR] of int: priority;
array[CR] of int: required_start_time;
array[CR] of int: required_duration;
array[CR] of int: ops_start_time;
array[CR] of int: ops_duration;

% ASSETS
array[INTELL,ASSET] of 0..1: sensor; % this can be modeled as an array of assets with allocated sensor but an asset can have multiple sensors
array[ASSET] of int: capacity;
array[ASSET] of int: on_station;
array[ASSET] of BASE: home;

% check the quality of input data

% A1 - check that the required time interval contains ops interval
constraint forall(i in CR)(assert((required_start_time[i] <= ops_start_time[i] /\
                                  required_duration[i] >= ops_duration[i]),
                                  "Ops time interval outside of Required time interval contains"));



% DECISION

%array[CR,ASSET] of var int: start_collection;
%array[CR,ASSET] of var int: collection_interval;

% annother way to formulate this is

array[CR] of var ASSET: ctl;
array[CR] of var int: allocated_collection_start;
array[CR] of var int: allocated_collection_duration;

% will try to use channeling between CRs and ASSETs


% CONSTRAINTS

include "alldifferent_except.mzn"; % not used in the eneumeration formulation; requires int values

% C1 - required intell matches the sensor collection
constraint forall(i in CR)(if ctl[i] != NO_ASSET then sensor[required_int[i],ctl[i]] = 1 endif);


% C2 - no overlapping of collection tasks
constraint forall(i,j in CR where i < j /\ ctl[i] = ctl[j]) (
            required_start_time[i] + required_duration[i] <= required_start_time[j] \/
            required_start_time[j] + required_duration[j] <= required_start_time[i]
        );

% C3 - only one asset per CR
constraint alldifferent_except(ctl, {NO_ASSET});

% C4 - check asset capacity
constraint forall(i in ASSET where i != NO_ASSET)(sum(j in CR where ctl[j] = i)(allocated_collection_duration[j]) <= capacity[i]);

% C5 - allocated period should cover at least ops time

constraint forall(i in CR)(if ctl[i] != NO_ASSET then
                             ops_start_time[i] >= allocated_collection_start[i] /\
                             ops_start_time[i] + ops_duration[i] <= allocated_collection_start[i] + allocated_collection_duration[i] endif);

% C6 - allocated time should be after asset on station time
% C7 - allocated time should be after required start time
% C8 - allocated duration should be less than required time

constraint forall(i in CR)(if ctl[i] != NO_ASSET then
                             allocated_collection_start[i] >= on_station[ctl[i]] /\
                             allocated_collection_start[i] >= required_start_time[i] /\
                             allocated_collection_start[i] + allocated_collection_duration[i] <=
                             required_start_time[i] + required_duration[i] endif);



% OBJECTIVE

% O1 - maximize the weighted, total collection time - same as O2

var int: obj_1 = sum(i in CR where ctl[i] != NO_ASSET)
                             (priority[i]);

%var int: obj_1 = sum(i in CR where ctl[i] != NO_ASSET)
%                             (priority[i] * allocated_collection_duration[i]);

% O2 - maximize overlaping of required collection with allocated collection

%var int: obj_2 = sum(i in CR where ctl[i] != NO_ASSET)
%                             (allocated_collection_duration[i] - required_duration[i]);

% O3 - penalty if CR has no allocated ASSET

var int: obj_3 = sum(i in CR)(ctl[i] = NO_ASSET);

var int: obj = obj_1 - obj_3;

solve maximize obj;
