function [LC_EUR_per_t, LCtrans_EUR_per_t, LCcomp_EUR_per_t] = LevelizedCosts(Ipipe_EUR, OCpipe_EUR_per_y, ...
    Icomp_EUR, OCcomp_EUR_per_y, ECcomp_EUR_per_y, ...
    Ipump_EUR, OCpump_EUR_per_y, ECpump_EUR_per_y, ...
    r, z_pipe, z_pumpcomp, m_kg_per_s, H_h)
%Equation (B.24) in Supplementary Material
%OUTPUT: LC_EUR_per_t = levelized costs of transport and conditioning [EUR.t-1]
%        LCtrans_EUR_per_t = levelized costs of transport [EUR.t-1]
%        LCcomp_EUR_per_t = levelized costs of conditioning [EUR.t-1]
%INPUT: Ipipe_EUR = investment costs for the pipe [EUR]
%       OCpipe_EUR_per_y = operating costs of the pipe [EUR.y-1]
%       Icompressor_EUR = investment costs of compressor [EUR]
%       OCcomp_EUR_per_y = operating costs of the compressor [EUR.y-1]
%       ECcomp_EUR_per_y = energy costs of compressor [EUR.y-1]
%       Ipump_EUR = investment costs of pumping stations [EUR]
%       OCpump_EUR_per_y = operating costs of the pump [EUR.y-1]
%       ECpump_EUR_per_y = energy costs of pumping stations [EUR.y-1]
%       r = discount rate [-]
%       z_pipe = lifetime of pipe [y]
%       z_pumpcomp = lifetime of pumps & compressors [y]
%       m_kg_per_s = CO2 mass flow [kg.s-1]
%       H_h = operating hours within a year [h]

CRFpipe = CapitalRecoveryFactor(r,z_pipe);
CRFpumpcomp = CapitalRecoveryFactor(r,z_pumpcomp);

LC_EUR_per_t = (CRFpipe.*Ipipe_EUR + CRFpumpcomp.*(Icomp_EUR + Ipump_EUR) + ...
    OCpipe_EUR_per_y + OCcomp_EUR_per_y + OCpump_EUR_per_y + ECcomp_EUR_per_y + ...
    ECpump_EUR_per_y)./(m_kg_per_s.*H_h.*3.6);

LCtrans_EUR_per_t = (CRFpipe.*Ipipe_EUR + CRFpumpcomp.*Ipump_EUR + ...
    OCpipe_EUR_per_y + OCpump_EUR_per_y + ECpump_EUR_per_y)./(m_kg_per_s.*H_h.*3.6);

LCcomp_EUR_per_t = (CRFpumpcomp.*Icomp_EUR + OCcomp_EUR_per_y + ECcomp_EUR_per_y)./(m_kg_per_s.*H_h.*3.6);

end