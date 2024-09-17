% Synthetically generated actions input class

classdef SynthActInput < SynthActions
    
    % Public constants
    properties (Constant)
        Classifs = createEnum({'movements','actions'}) % classifications enumeration
        AllLbls = {SynthActions.MovementLbls,SynthActions.ActionLbls} % cell with all labels enumerations
        Features = true % handcrafted features data
    end
    
    % Public constants
    properties (SetAccess = protected)
        AllNLbls = InputData.setAllNLbls(SynthActInput.AllLbls)
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
            lbls = zeros(length(SynthActInput.AllLbls),0);
            for i = 1:obj.Data.traj_p.plans_scen
                action = obj.generateAction(objtypes,pos);
                for j = 1:length(action.movements)
                    [traj,pos,lbls] = obj.generateMovementTraj(action,j,traj,pos,lbls);
                end
                [traj,lbls] = obj.generateWaitTraj(traj,pos,lbls);
            end
            obj.VisStt.traj = traj;
        end
        % Generate new action
        function action = generateAction(obj,objtypes,pos)
            tgt_obj = randi(obj.Data.n_objects);
            tgt_x = 1 + round(rand*obj.Data.dims.scen_w);
            tgt_y = 1 + round(rand*obj.Data.dims.scen_h);
            action = obj.generateAction@SynthActions(objtypes,pos,tgt_obj,tgt_x,tgt_y);
        end
    end
    
end