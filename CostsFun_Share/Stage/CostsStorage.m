function [storage] = CostsStorage(storage, i)
%This function calculates the costs for storage
%INPUT: storage = struct containing the storage information
%       i = plant index
%OUTPUT: storage = struct containing the storage information
%        LC_CHF_per_t = levelized costs [CHF/t]


%The amount arriving at the storage will vary depending on the transport
%mode
% m_capt_t_per_y = plant.capture(i).m_capt_t_per_y; %[t/y]

for st = 1:size(storage.start,2)
    for go = 1:size(storage.start(st).goal,2)

        C_TransportStorage_CHF_per_t = storage.start(st).goal(go).C_TransportStorage_CHF_per_t; %[CHF/t]

        %% Calculs

        LC_CHF_per_t = C_TransportStorage_CHF_per_t; %[CHF/t]

        %% Export

        storage.capture(i).start(st).goal(go).costs.LC_CHF_per_t = LC_CHF_per_t;
        storage.capture(i).start(st).goal(go).costs.TextLegend = 'Storage';
        
    end
end

end

