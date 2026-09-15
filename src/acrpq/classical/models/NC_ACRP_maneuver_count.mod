# Alternative objective layered on top of the historical Code/NC_ACRP.mod.
#
# The historical variables, bounds and separation constraints are not copied or
# changed.  This module only introduces one binary indicator per aircraft and
# links it exactly to the original continuous controls:
#   moved[i] = 0  =>  q[i] = 1 and theta[i] = 0
#   moved[i] = 1  =>  the historical control bounds remain available.
# At an optimum, moved[i] cannot remain 1 gratuitously because it has unit cost.

var moved{i in A} binary;

minimize ManeuverCount: sum{i in A} moved[i];

s.t. LinkThetaUpper{i in A}: theta[i] <= hmax * moved[i];
s.t. LinkThetaLower{i in A}: theta[i] >= hmin * moved[i];
s.t. LinkSpeedUpper{i in A}: q[i] - 1 <= (qmax - 1) * moved[i];
s.t. LinkSpeedLower{i in A}: 1 - q[i] <= (1 - qmin) * moved[i];
