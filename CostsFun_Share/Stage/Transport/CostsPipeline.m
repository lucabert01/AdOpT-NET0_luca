function [pipeline] = CostsPipeline(pipeline, conditioning, finance, i)
%This function calculates the costs for the pipeline transport in gaseous
%phase.
%INPUT: pipeline = struct containing the pipeline information
%       conditioning = struct containing the conditioning information
%       i = plant index
%OUTPUT: pipeline = struct containing the pipeline information
%           C_pipeline_CHF_per_t_per_km = unitary cost for the transport by pipeline [CHF/t/km]
%           C_pipeline_CHF_per_y = yearly costs for the pipeline [CHF/y]
%           TAC_CHF_per_y = total annualized costs [CHF/y]
%           LC_CHF_per_t = levelized costs [CHF/t]

%% Data

m_comp_t_per_y = conditioning.capture(i).m_comp_t_per_y; %[t/y]
m0_t_per_y = 1; %[t/y]
d0_km = 1;
    
for st = 1:size(pipeline.start,2)
    for go = 1:size(pipeline.start(st).goal,2)

%         d_km = pipeline.start(st).goal(go).d_km; %[km]
        d_On_km = pipeline.start(st).goal(go).d_On_km;
        d_Off_km = pipeline.start(st).goal(go).d_Off_km;

        cOn_CHF_per_t_per_km = (pipeline.a1_on + pipeline.a2_on.*(d_On_km./d0_km).^pipeline.a3_on.*(m_comp_t_per_y./m0_t_per_y).^pipeline.a4_on).*finance.x_EURCHF;
        cOff_CHF_per_t_per_km = (pipeline.a1_off + pipeline.a2_off.*(d_On_km./d0_km).^pipeline.a3_off.*(m_comp_t_per_y./m0_t_per_y).^pipeline.a4_off).*finance.x_EURCHF;

        %% Calculs

        C_on_CHF_per_y = cOn_CHF_per_t_per_km.*m_comp_t_per_y.*d_On_km; %[CHF/y]
        C_off_CHF_per_y = cOff_CHF_per_t_per_km.*m_comp_t_per_y.*d_Off_km; %[CHF/y]

        C_pipeline_CHF_per_y = C_on_CHF_per_y + C_off_CHF_per_y; %[CHF/y]

        TAC_CHF_per_y = C_pipeline_CHF_per_y; %[CHF/y]
        LC_CHF_per_t = TAC_CHF_per_y./m_comp_t_per_y; %[CHF/t]

        pipeline.capture(i).start(st).goal(go).C_pipelineOn_CHF_per_t_per_km = cOn_CHF_per_t_per_km; %[CHF/t/km]
        pipeline.capture(i).start(st).goal(go).C_pipelineOff_CHF_per_t_per_km = cOff_CHF_per_t_per_km; %[CHF/t/km]
        pipeline.capture(i).start(st).goal(go).costs.C_pipeline_CHF_per_y = C_pipeline_CHF_per_y;
        pipeline.capture(i).start(st).goal(go).costs.TAC_CHF_per_y = TAC_CHF_per_y;
        pipeline.capture(i).start(st).goal(go).costs.LC_CHF_per_t = LC_CHF_per_t;
        
        %Resize
    
        fn = fieldnames(pipeline.capture(i).start(st).goal(go).costs);
        
        for j = 1:numel(fn)
            if size(pipeline.capture(i).start(st).goal(go).costs.(fn{j}),1) == 1 && size(pipeline.capture(i).start(st).goal(go).costs.(fn{j}),2) == 3
                pipeline.capture(i).start(st).goal(go).costs.(fn{j}) = pipeline.capture(i).start(st).goal(go).costs.(fn{j}).*ones(3,1);
            elseif size(pipeline.capture(i).start(st).goal(go).costs.(fn{j}),1) == 3 && size(pipeline.capture(i).start(st).goal(go).costs.(fn{j}),2) == 1
                pipeline.capture(i).start(st).goal(go).costs.(fn{j}) = pipeline.capture(i).start(st).goal(go).costs.(fn{j}).*ones(1,3);
            elseif size(pipeline.capture(i).start(st).goal(go).costs.(fn{j}),1) == 1 && size(pipeline.capture(i).start(st).goal(go).costs.(fn{j}),2) == 1
                pipeline.capture(i).start(st).goal(go).costs.(fn{j}) = pipeline.capture(i).start(st).goal(go).costs.(fn{j}).*ones(3,3);
            end
        end

    end
end

end

