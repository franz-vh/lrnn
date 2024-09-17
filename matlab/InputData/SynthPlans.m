% Abstract class from which all inputs involving synthetic plans inherit

classdef SynthPlans < SynthActions
    
    % Public constants
    properties (Constant)
        Classifs = createEnum({'movements','actions','plans'}) % classifications enumeration
        PlanLbls = createEnum({'horizontal','vertical','square','triangle','random'}) % plan labels enumeration
        AllLbls = {SynthActions.MovementLbls,SynthActions.ActionLbls,SynthPlans.PlanLbls} % cell with all labels enumerations
    end
    
    % Public constants
    properties (SetAccess = protected)
        AllNLbls = InputData.setAllNLbls(SynthPlans.AllLbls)
    end
    
    % Sample generation methods
    methods (Access = protected)
        % Current scenario state
        function [objtypes,traj,lbls] = getTrajectories(obj)
            [objtypes,pos] = obj.generateScenario();
            traj.x = [];
            traj.y = [];
            traj.x_hand = [];
            traj.y_hand = [];
            traj.hand_h = [];
            lbls = zeros(length(SynthPlanInput.AllLbls),0);
            for i = 1:obj.Data.traj_p.plans_scen
                plan = obj.generatePlan(objtypes,pos);
                for j = 1:length(plan.actions)
                    for k = 1:length(plan.actions{j}.movements)
                        [traj,pos,lbls] = obj.generateMovementTraj(plan,j,k,traj,pos,lbls);
                    end
                end
                [traj,lbls] = obj.generateWaitTraj(traj,pos,lbls,plan.id);
            end
            obj.VisStt.traj = traj;
        end
        % Generate new plan
        function plan = generatePlan(obj,objtypes,pos)
            plan.id = randi(length(fieldnames(SynthPlanInput.PlanLbls)));
            switch plan.id
                case SynthPlanInput.PlanLbls.horizontal
                    tgt_x = round(linspace(1/(2*obj.Data.n_objects),1-1/(2*obj.Data.n_objects),obj.Data.n_objects)*obj.Data.dims.scen_w)';
                    tgt_y = (.8*rand+.1)*obj.Data.dims.scen_h*ones(obj.Data.n_objects,1);
                case SynthPlanInput.PlanLbls.vertical
                    tgt_x = (.8*rand+.1)*obj.Data.dims.scen_w*ones(obj.Data.n_objects,1);
                    tgt_y = round(linspace(1/(2*obj.Data.n_objects),1-1/(2*obj.Data.n_objects),obj.Data.n_objects)*obj.Data.dims.scen_h)';
                case SynthPlanInput.PlanLbls.square
                    rot = pi/2*rand;
                    tgt_x0 = round([1/4;1/4;3/4;3/4]*obj.Data.dims.scen_w) - obj.Data.dims.scen_w/2;
                    tgt_y0 = round([1/4;3/4;1/4;3/4]*obj.Data.dims.scen_h) - obj.Data.dims.scen_h/2;
                    tgt_x = tgt_x0*cos(rot) - tgt_y0*sin(rot) + obj.Data.dims.scen_w/2;
                    tgt_y = tgt_x0*sin(rot) + tgt_y0*cos(rot) + obj.Data.dims.scen_h/2;
                case SynthPlanInput.PlanLbls.triangle
                    rot = 2*pi/3*rand;
                    tgt_x0 = round([1/2;1/2-sqrt(3)/8;1/2+sqrt(3)/8;1/2]*obj.Data.dims.scen_w) - obj.Data.dims.scen_w/2;
                    tgt_y0 = round([1/4;5/8;5/8;1/2]*obj.Data.dims.scen_h) - obj.Data.dims.scen_h/2;
                    tgt_x = tgt_x0*cos(rot) - tgt_y0*sin(rot) + obj.Data.dims.scen_w/2;
                    tgt_y = tgt_x0*sin(rot) + tgt_y0*cos(rot) + obj.Data.dims.scen_h/2;
                case SynthPlanInput.PlanLbls.random
                    tgt_x = round(rand(obj.Data.n_objects,1)*obj.Data.dims.scen_w);
                    tgt_y = round(rand(obj.Data.n_objects,1)*obj.Data.dims.scen_h);
            end
            tgt_x = tgt_x(1:min(length(tgt_x),obj.Data.n_objects));
            tgt_y = tgt_y(1:min(length(tgt_y),obj.Data.n_objects));
            tgt_x = [tgt_x;ones(obj.Data.n_objects-length(tgt_x),1)];
            tgt_y = [tgt_y;ones(obj.Data.n_objects-length(tgt_y),1)];
            actions = cell(1,obj.Data.n_objects);
            for i = 1:obj.Data.n_objects
                [~,tgt_obj] = min((pos.x_hand-pos.x).^2 + (pos.y_hand-pos.y).^2);
                [~,tgt_id] = min((pos.x(tgt_obj)-tgt_x).^2 + (pos.y(tgt_obj)-tgt_y).^2);
                actions{i} = obj.generateAction(objtypes,pos,tgt_obj,tgt_x(tgt_id),tgt_y(tgt_id));
                pos.x(actions{i}.obj) = Inf;
                pos.y(actions{i}.obj) = Inf;
                tgt_x(tgt_id) = -Inf;
                tgt_y(tgt_id) = -Inf;
                pos.x_hand = actions{i}.x2;
                pos.y_hand = actions{i}.y2;   
            end
            plan.actions = actions;
        end
        % Generate movement trajectory
        function [traj,pos,lbls] = generateMovementTraj(obj,plan,action,movement,traj,pos,lbls)
            [traj,pos,newlbls] = generateMovementTraj@SynthActions(obj,plan.actions{action},movement,traj,pos,[]);
            lbls = [lbls,[newlbls;plan.id*ones(1,size(newlbls,2))]];
        end
        % Generate wait trajectory
        function [traj,lbls] = generateWaitTraj(obj,traj,pos,lbls,plan_id)
            [traj,newlbls] = generateWaitTraj@SynthActions(obj,traj,pos,[]);
            lbls = [lbls,[newlbls;plan_id*ones(1,size(newlbls,2))]];
        end
    end
    
end