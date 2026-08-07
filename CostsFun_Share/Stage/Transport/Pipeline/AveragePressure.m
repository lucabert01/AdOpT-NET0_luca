function [Pave_Pa] = AveragePressure(Poutlet_Pa, Pinlet_Pa)
%Equation (B.3) in Supplementary Material
%OUTPUT: Pave_Pa = average pressure [Pa]
%INPUT: Poutlet_Pa = outlet pressure [Pa]
%       Pinlet_Pa = inlet pressure [Pa]

Pave_Pa = 2.*(Poutlet_Pa + Pinlet_Pa - Poutlet_Pa.*Pinlet_Pa./(Poutlet_Pa + Pinlet_Pa))./3;

end