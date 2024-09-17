% Abstract class from which all inputs involving synthetic actions inherit

classdef SynthActions < InputData
    
    % Public constants
    properties (Constant)
        MovementLbls = createEnum({'move','pick','carry','place','wait'}) % movement labels enumeration
        ActionLbls = createEnum({'picknplace','push','pull','wait'}) % action labels enumeration
        Range = [0,1]
    end

    % Public constants (abstract)
    properties (Abstract,Constant)
        Features {mustBeA(Features,"logical")} % raw or handcrafted features data
    end
    
    % Data and state variables required for scenario visualization
    properties (SetAccess = protected)
        VisData
        VisStt
    end
    
    % Main public methods
    methods
        % Constructor
        function obj = SynthActions(~)
            obj.initData();
            obj.initVis();
            obj.hardRestart(struct('asnewobject',true));
        end
        % Get next batch of samples
        function [batch,lbls] = getBatch(obj,batchsize)
            [batch,lbls] = getBatch@InputData(obj,batchsize);
            obj.updateVisStt();
        end
        % Get next batch of samples and internal states
        function [batch,lbls,stts] = getStateBatch(obj,batchsize)
            batch = zeros(obj.Size,batchsize);
            lbls = zeros(length(obj.AllLbls),batchsize);
            stts = cell(1,batchsize);
            nxtsmpls = obj.Stt.nxtsmpls;
            nxtlbls = obj.Stt.nxtlbls;
            i1 = 1;
            i2 = i1 + size(nxtsmpls,2);
            while i2 <= batchsize
                batch(:,i1:i2-1) = nxtsmpls;
                lbls(:,i1:i2-1) = nxtlbls;
                nxtstts = cell(1,size(nxtsmpls,2));
                for i = 1:size(nxtstts,2)
                    nxtstts{i}.objtypes = obj.VisStt.objtypes;
                    nxtstts{i}.x = obj.VisStt.traj.x(:,i);
                    nxtstts{i}.y = obj.VisStt.traj.y(:,i);
                    nxtstts{i}.x_hand = obj.VisStt.traj.x_hand(:,i);
                    nxtstts{i}.y_hand = obj.VisStt.traj.y_hand(:,i);
                    nxtstts{i}.hand_h = obj.VisStt.traj.hand_h(:,i);
                end
                stts(:,i1:i2-1) = nxtstts;
                [nxtsmpls,nxtlbls] = obj.getSequence();
                i1 = i2;
                i2 = i1 + size(nxtsmpls,2);
            end
            batch(:,i1:end) = nxtsmpls(:,1:batchsize-i1+1);
            lbls(:,i1:end) = nxtlbls(:,1:batchsize-i1+1);
            nxtstts = cell(1,batchsize-i1+1);
            for i = 1:size(nxtstts,2)
                nxtstts{i}.objtypes = obj.VisStt.objtypes;
                nxtstts{i}.x = obj.VisStt.traj.x(:,i);
                nxtstts{i}.y = obj.VisStt.traj.y(:,i);
                nxtstts{i}.x_hand = obj.VisStt.traj.x_hand(:,i);
                nxtstts{i}.y_hand = obj.VisStt.traj.y_hand(:,i);
                nxtstts{i}.hand_h = obj.VisStt.traj.hand_h(:,i);
            end
            stts(:,i1:end) = nxtstts;
            obj.Stt.nxtsmpls = nxtsmpls(:,batchsize-i1+2:end);
            obj.Stt.nxtlbls = nxtlbls(:,batchsize-i1+2:end);
            obj.updateVisStt();
        end
        % Restart state
        function restart(obj,~)
            obj.Stt.nxtsmpls = [];
            obj.Stt.nxtlbls = [];
        end
        % Hard restart (new object types)
        function hardRestart(obj,params)
            if exist('params','var') && isfield(params,'asnewobject') && params.asnewobject
                obj.generateObjectTypeIds();
                obj.generateAffordances();
            end
            obj.restart();
        end
    end
    
    % Data initialization methods
    methods (Access = protected)
        % Input data initialization
        function initData(obj)
            % Objects and object types
            data.n_objtypes = 5;
            data.in_p.objid_sz = 10;
            data.objid_actv = 3;
            data.n_objects = 4;
            % Scenario and object dimensions
            data.dims.scen_h = 1000;
            data.dims.scen_w = 1000;
            data.dims.obj_h = 50;
            data.dims.obj_w = 50;
            data.dims.ophand_h = 100;
            data.dims.clhand_h = 50;            
            % Trajectory generation parameters
            data.traj_p.fs = 30;
            data.traj_p.plans_scen = 5;
            data.traj_p.amin = 5000;
            data.traj_p.amax = 10000;
            data.traj_p.hand_sigma = min(data.dims.obj_h,data.dims.obj_w)/4;
            data.traj_p.obj_sigma = min(data.dims.scen_h,data.dims.scen_w)/10;
            data.traj_p.vxymax = sqrt(3/8*data.traj_p.amax*max(data.dims.scen_w,data.dims.scen_h));
            data.traj_p.tmax_wait = 1;
            data.traj_p.tmin_cl = 0.05;
            data.traj_p.tmax_cl = 0.1;
            data.traj_p.tmin_op = 0.2;
            data.traj_p.tmax_op = 0.5;
            % Attention parameters
            data.att_p.A_factor = 1.5;
            data.att_p.x0_factor = 1/4;
            data.att_p.y0_factor = 1/4;
            data.att_p.sigma_X_min = 30;
            data.att_p.sigma_X_factor = 1/8;
            data.att_p.sigma_Y_min = 30;
            data.att_p.sigma_Y_factor = 1/32;
            data.att_p.v_factor = 0.7;
            % Input parameters
            data.in_p.opcl_sz = 2;
            data.in_p.v_sd = 3;
            data.in_p.pos_sd = 5;
            data.in_p.v_nthfactor = 2;
            v_sigma = 1/2;
            data.in_p.v_var = v_sigma^2;
            data.in_p.pos_nthfactor = 16;
            pos_sigma = 1/4;
            data.in_p.pos_var = pos_sigma^2;
            % Input generation constants
            data.in_g.opcl_ref = linspace(data.dims.clhand_h,data.dims.ophand_h,data.in_p.opcl_sz)';
            [Xv,Yv] = meshgrid(-1:2/(data.in_p.v_sd-1):1,-1:2/(data.in_p.v_sd-1):1);
            data.in_g.Xv = reshape(Xv,1,1,numel(Xv));
            data.in_g.Yv = reshape(Yv,1,1,numel(Yv));
            data.in_g.Xv_hand = Xv(:);
            data.in_g.Yv_hand = Yv(:);
            [Xpos,Ypos] = meshgrid(-1:2/(data.in_p.pos_sd-1):1,-1:2/(data.in_p.pos_sd-1):1);
            data.in_g.Xpos = reshape(Xpos,1,1,numel(Xpos));
            data.in_g.Ypos = reshape(Ypos,1,1,numel(Ypos));
            % Input parameters
            data.in_p.sigma_n = 0.02;
            if obj.Features
                obj.Size = data.in_p.objid_sz + data.in_p.opcl_sz + 2*data.in_p.v_sd^2 + data.in_p.pos_sd^2;
            else
                obj.Size = 3 + data.n_objects*(data.in_p.objid_sz+2);
            end
            obj.Dims = obj.Size;
            obj.Data = data;
        end
        % Random sparse id generation for a given number of object types
        function generateObjectTypeIds(obj)
            obj.Data.objtype_ids = [rand(obj.Data.n_objtypes,obj.Data.objid_actv),zeros(obj.Data.n_objtypes,obj.Data.in_p.objid_sz-obj.Data.objid_actv)];
            for i = 1:obj.Data.n_objtypes
                obj.Data.objtype_ids(i,randperm(obj.Data.in_p.objid_sz)) = obj.Data.objtype_ids(i,:);
            end
        end
        % Random affordance generation for a given number of object types
        function generateAffordances(obj)
            n_affordances = length(fieldnames(SynthActions.ActionLbls)) - 1;
            obj.Data.affordances = randi([0,1],obj.Data.n_objtypes,n_affordances);
            while ~(all(any(obj.Data.affordances,1)) && all(any(obj.Data.affordances,2)))
                obj.Data.affordances = randi([0,1],obj.Data.n_objtypes,n_affordances);
            end
        end
        % Visualization parameters initialization
        function initVis(obj)
            obj.VisData.hand_w = 70;
            obj.VisData.margin = 200;
            obj.VisData.colornames = {'red','green','blue','yellow','magenta','cyan','gray'};
            obj.VisData.colors = [1,0,0;0,1,0;0,0,1;1,1,0;1,0,1;0,1,1;.5,.5,.5];
            color_objs = permute(obj.VisData.colors,[4,3,2,1]);
            obj.VisData.color_objs = ones(obj.Data.dims.obj_h,obj.Data.dims.obj_w).*color_objs;
            [obj.VisData.Xscenario,obj.VisData.Yscenario] = meshgrid(1:obj.Data.dims.scen_w+2*obj.VisData.margin,1:obj.Data.dims.scen_h+2*obj.VisData.margin);
        end
    end
    
    % Sample generation methods
    methods (Access = protected)
        % Get sample sequence
        function [samples,lbls] = getSequence(obj)
            [objtypes,traj,lbls] = obj.getTrajectories();
            samples = obj.getSamplesFromTraj(objtypes,traj);
            samples = SynthActions.addNoise(samples,obj.Data.in_p.sigma_n);
            if obj.Features
                samples = SynthActions.smoothSamples(samples);
                vel = obj.estimateVelocities(samples);
                attention = obj.getAttentions(samples,vel);
                samples = obj.generateInput(samples,vel,attention);
            end
        end
        % Generate new scenario
        function [objtypes,pos] = generateScenario(obj)
            objtypes = randi(obj.Data.n_objtypes,1,obj.Data.n_objects);
            obj.VisStt.objtypes = objtypes;
            pos.x_hand = 1 + round(rand*obj.Data.dims.scen_w);
            pos.y_hand = 1 + round(rand*obj.Data.dims.scen_h);
            pos.hand_h = obj.Data.dims.clhand_h + rand*(obj.Data.dims.ophand_h-obj.Data.dims.clhand_h);
            pos.x = 1 + round(rand(obj.Data.n_objects,1)*(obj.Data.dims.scen_w-1));
            pos.y = 1 + round(rand(obj.Data.n_objects,1)*(obj.Data.dims.scen_h-1));
            %figure(1),uitable('Data', color_strs(objects_types)', 'ColumnName', 'object types','Position',[300 120 220 150]);
        end
        % Generate new action
        function action = generateAction(obj,objtypes,pos,tgt_obj,tgt_x,tgt_y)
            cur_affordances = obj.Data.affordances(objtypes,:);
            action.obj = tgt_obj;
            action_ids = find(cur_affordances(action.obj,:));
            action.id = action_ids(randi(length(action_ids)));
            switch action.id
                case SynthActions.ActionLbls.picknplace
                    action.movements = [SynthActions.MovementLbls.move,SynthActions.MovementLbls.pick,SynthActions.MovementLbls.carry,SynthActions.MovementLbls.place];
                    shift_x = 0;
                    shift_y = 0;
                case SynthActions.ActionLbls.push
                    action.movements = [SynthActions.MovementLbls.move,SynthActions.MovementLbls.carry];
                    distance = sqrt((tgt_x-pos.x(action.obj))^2 + (tgt_y-pos.y(action.obj))^2) + 0.0001;
                    shift_x = mean(obj.Data.dims.obj_h,obj.Data.dims.obj_w)*(pos.x(action.obj)-tgt_x)/distance;
                    shift_y = mean(obj.Data.dims.obj_h,obj.Data.dims.obj_w)*(pos.y(action.obj)-tgt_y)/distance;
                case SynthActions.ActionLbls.pull
                    action.movements = [SynthActions.MovementLbls.move,SynthActions.MovementLbls.carry];
                    distance = sqrt((tgt_x-pos.x(action.obj))^2 + (tgt_y-pos.y(action.obj))^2) + 0.0001;
                    shift_x = mean(obj.Data.dims.obj_h,obj.Data.dims.obj_w)*(tgt_x-pos.x(action.obj))/distance;
                    shift_y = mean(obj.Data.dims.obj_h,obj.Data.dims.obj_w)*(tgt_y-pos.y(action.obj))/distance;
            end
            action.x1 = pos.x(action.obj) + shift_x;
            action.y1 = pos.y(action.obj) + shift_y;
            action.x2 = tgt_x + shift_x;
            action.y2 = tgt_y + shift_y;
        end
        % Generate movement trajectory
        function [traj,pos,lbls] = generateMovementTraj(obj,action,movement,traj,pos,lbls)
            %v = vmin + rand*(vmax-vmin);
            a = obj.Data.traj_p.amin + rand*(obj.Data.traj_p.amax-obj.Data.traj_p.amin);
            switch action.movements(movement)
                case SynthActions.MovementLbls.move
                    x1 = action.x1 + obj.Data.traj_p.hand_sigma*randn;
                    y1 = action.y1 + obj.Data.traj_p.hand_sigma*randn;
                    distance = sqrt((x1-pos.x_hand)^2 + (y1-pos.y_hand)^2) + 0.0001;
                    t = [1/obj.Data.traj_p.fs 2/obj.Data.traj_p.fs:1/obj.Data.traj_p.fs:sqrt(6*distance/a)];
                    a = 6*distance/t(end)^2;
                    s = (1/2*a*t.^2 - 1/3*a*t.^3/t(end))/distance;
                    x_hand = pos.x_hand + (x1-pos.x_hand)*s;
                    y_hand = pos.y_hand + (y1-pos.y_hand)*s;
                    hand_h = pos.hand_h*ones(size(t));
                    x_obj = pos.x(action.obj)*ones(size(t));
                    y_obj = pos.y(action.obj)*ones(size(t));
                    if action.id == SynthActions.ActionLbls.picknplace
                        t_open = obj.Data.traj_p.tmin_op + rand*(obj.Data.traj_p.tmax_op-obj.Data.traj_p.tmin_op);
                        t_open = min(max(t_open,1/obj.Data.traj_p.fs),t(end));
                        n_open = round(t_open*obj.Data.traj_p.fs);
                        hand_h(end-n_open+1:end) = linspace(pos.hand_h,obj.Data.dims.ophand_h,n_open);
                    end
                case SynthActions.MovementLbls.pick
                    t_grip = obj.Data.traj_p.tmin_cl + rand*(obj.Data.traj_p.tmax_cl-obj.Data.traj_p.tmin_cl);
                    t = [1/obj.Data.traj_p.fs 2/obj.Data.traj_p.fs:1/obj.Data.traj_p.fs:t_grip];
                    x_hand = pos.x_hand*ones(size(t));
                    y_hand = pos.y_hand*ones(size(t));
                    hand_h = pos.hand_h + (obj.Data.dims.clhand_h-pos.hand_h)*t/t(end);
                    x_obj = pos.x(action.obj)*ones(size(t));
                    y_obj = pos.y(action.obj)*ones(size(t));
                case SynthActions.MovementLbls.carry
                    x2 = action.x2 + obj.Data.traj_p.hand_sigma*randn;
                    y2 = action.y2 + obj.Data.traj_p.hand_sigma*randn;
                    distance = sqrt((x2-pos.x_hand)^2 + (y2-pos.y_hand)^2) + 0.0001;
                    t = [1/obj.Data.traj_p.fs 2/obj.Data.traj_p.fs:1/obj.Data.traj_p.fs:sqrt(6*distance/a)];
                    a = 6*distance/t(end)^2;
                    s = (1/2*a*t.^2 - 1/3*a*t.^3/t(end))/distance;
                    x_hand = pos.x_hand + (x2-pos.x_hand)*s;
                    y_hand = pos.y_hand + (y2-pos.y_hand)*s;
                    hand_h = pos.hand_h*ones(size(t));
                    x_obj = pos.x(action.obj) + (x2-pos.x_hand)*s;
                    y_obj = pos.y(action.obj) + (y2-pos.y_hand)*s;
                case SynthActions.MovementLbls.place
                    t_grip = obj.Data.traj_p.tmin_cl + rand*(obj.Data.traj_p.tmax_cl-obj.Data.traj_p.tmin_cl);
                    t = [1/obj.Data.traj_p.fs 2/obj.Data.traj_p.fs:1/obj.Data.traj_p.fs:t_grip];
                    x_hand = pos.x_hand*ones(size(t));
                    y_hand = pos.y_hand*ones(size(t));
                    hand_hf = pos.hand_h + rand*(obj.Data.dims.ophand_h-pos.hand_h);
                    hand_h = pos.hand_h + (hand_hf-pos.hand_h)*t/t(end);
                    x_obj = pos.x(action.obj)*ones(size(t));
                    y_obj = pos.y(action.obj)*ones(size(t));
            end
            x = pos.x.*ones(size(t));
            y = pos.y.*ones(size(t));
            x(action.obj,:) = x_obj;
            y(action.obj,:) = y_obj;
            traj.x = [traj.x,x];
            traj.y = [traj.y,y];
            traj.x_hand = [traj.x_hand,x_hand];
            traj.y_hand = [traj.y_hand,y_hand];
            traj.hand_h = [traj.hand_h,hand_h];
            pos.x = traj.x(:,end);
            pos.y = traj.y(:,end);
            pos.x_hand = traj.x_hand(:,end);
            pos.y_hand = traj.y_hand(:,end);
            pos.hand_h = traj.hand_h(:,end);
            lbls = [lbls,[action.movements(movement);action.id].*ones(size(t))];
        end
        % Generate wait trajectory
        function [traj,lbls] = generateWaitTraj(obj,traj,pos,lbls)
            nsamples = round(rand*obj.Data.traj_p.tmax_wait*obj.Data.traj_p.fs);
            traj.x_hand = [traj.x_hand,pos.x_hand*ones(1,nsamples)];
            traj.y_hand = [traj.y_hand,pos.y_hand*ones(1,nsamples)];
            traj.hand_h = [traj.hand_h,pos.hand_h*ones(1,nsamples)];
            traj.x = [traj.x,pos.x.*ones(1,nsamples)];
            traj.y = [traj.y,pos.y.*ones(1,nsamples)];
            lbls = [lbls,[SynthActions.MovementLbls.wait;SynthActions.ActionLbls.wait].*ones(1,nsamples)];
        end
        % Get samples from object types and trajectories
        function samples = getSamplesFromTraj(obj,objtypes,traj)
            object_ids = obj.Data.objtype_ids(objtypes,:)';
            object_ids = object_ids(:)*ones(1,size(traj.x,2));
            x_hand = traj.x_hand/obj.Data.dims.scen_w;
            y_hand = traj.y_hand/obj.Data.dims.scen_h;
            hand_h = (traj.hand_h-obj.Data.dims.clhand_h)/(obj.Data.dims.ophand_h-obj.Data.dims.clhand_h);
            x = traj.x/obj.Data.dims.scen_w;
            y = traj.y/obj.Data.dims.scen_h;
            samples = [object_ids;x_hand;y_hand;hand_h;x;y];
        end
        % Get object ids from samples
        function object_ids = getObjectIdsFromSamples(obj,samples)
            object_ids = samples(1:obj.Data.n_objects*obj.Data.in_p.objid_sz,:);
        end
        % Get x coordinate of hand from samples
        function x_hand = getXHandFromSamples(obj,samples)
            x_hand = samples(obj.Data.n_objects*obj.Data.in_p.objid_sz+1,:)*obj.Data.dims.scen_w;
        end
        % Get y coordinate of hand from samples
        function y_hand = getYHandFromSamples(obj,samples)
            y_hand = samples(obj.Data.n_objects*obj.Data.in_p.objid_sz+2,:)*obj.Data.dims.scen_h;
        end
        % Get x hand height from samples
        function hand_h = getHandHFromSamples(obj,samples)
            hand_h = samples(obj.Data.n_objects*obj.Data.in_p.objid_sz+3,:)*(obj.Data.dims.ophand_h-obj.Data.dims.clhand_h) + obj.Data.dims.clhand_h;
        end
        % Get x coordinate of objects from samples
        function x = getXFromSamples(obj,samples)
            sample0 = obj.Data.n_objects*obj.Data.in_p.objid_sz + 3;
            x = samples(sample0+1:sample0+obj.Data.n_objects,:)*obj.Data.dims.scen_w;
        end
        % Get y coordinate of objects from samples
        function y = getYFromSamples(obj,samples)
            sample0 = obj.Data.n_objects*obj.Data.in_p.objid_sz + obj.Data.n_objects + 3;
            y = samples(sample0+1:end,:)*obj.Data.dims.scen_h;
        end
        % Estimate velocity of hand and objects
        function vel = estimateVelocities(obj,samples)
            vel.x_hand = [zeros(1,1),diff(obj.getXHandFromSamples(samples),[],2)]*obj.Data.traj_p.fs;
            vel.y_hand = [zeros(1,1),diff(obj.getYHandFromSamples(samples),[],2)]*obj.Data.traj_p.fs;
            x = obj.getXFromSamples(samples);
            vel.x = [zeros(size(x,1),1),diff(x,[],2)]*obj.Data.traj_p.fs;
            y = obj.getYFromSamples(samples);
            vel.y = [zeros(size(y,1),1),diff(y,[],2)]*obj.Data.traj_p.fs;
        end
        % Estimate attention to each object
        function attention = getAttentions(obj,samples,vel)
            att_p = obj.Data.att_p;
            v_hand = sqrt(vel.x_hand.^2+vel.y_hand.^2);
            att_G.x0 = obj.getXHandFromSamples(samples) + att_p.x0_factor*vel.x_hand;
            att_G.y0 = obj.getYHandFromSamples(samples) + att_p.y0_factor*vel.y_hand;
            sigma_X = att_p.sigma_X_min + att_p.sigma_X_factor*v_hand;
            sigma_Y = att_p.sigma_Y_min + att_p.sigma_Y_factor*v_hand;
            theta = atan2(vel.y_hand,-vel.x_hand);
            att_G.A = att_p.A_factor*sqrt(att_p.sigma_X_min*att_p.sigma_Y_min./(sigma_X.*sigma_Y));
            att_G.a = cos(theta).^2./(2*sigma_X.^2) + sin(theta).^2./(2*sigma_Y.^2);
            att_G.b = -sin(2*theta)./(4*sigma_X.^2) + sin(2*theta)./(4*sigma_Y.^2);
            att_G.c = sin(theta).^2./(2*sigma_X.^2) + cos(theta).^2./(2*sigma_Y.^2);
            obj.VisStt.att_G = att_G;
            x = obj.getXFromSamples(samples);
            y = obj.getYFromSamples(samples);
            attention = att_G.A.*exp(-(att_G.a.*(x-att_G.x0).^2 + 2*att_G.b.*(x-att_G.x0).*(y-att_G.y0) + att_G.c.*(y-att_G.y0).^2));
            attention = attention + att_p.v_factor*sqrt(vel.x.^2+vel.y.^2)/obj.Data.traj_p.vxymax;
            att_sum = sum(attention,1);
            attention(:,att_sum > 1) = attention(:,att_sum > 1)./att_sum(att_sum > 1);
        end
        % Generate input data
        function input = generateInput(obj,samples,vel,attention)
            in_g = obj.Data.in_g;
            in_p = obj.Data.in_p;
            vxymax = obj.Data.traj_p.vxymax;
            att_object_id = reshape(sum(reshape(obj.getObjectIdsFromSamples(samples),[obj.Data.in_p.objid_sz,obj.Data.n_objects,size(attention,2)]).*reshape(attention,[1,obj.Data.n_objects,size(attention,2)]),2),[obj.Data.in_p.objid_sz,size(attention,2)]);
            open_close = 1 - abs(obj.getHandHFromSamples(samples)-in_g.opcl_ref)/(in_g.opcl_ref(2)-in_g.opcl_ref(1));
            open_close(open_close < 0) = 0;
            att_object_v = reshape(sum(exp(-1/(2*in_p.v_var)*((in_g.Xv-SynthActions.nTanh(vel.x/vxymax,in_p.v_nthfactor)).^2 + (in_g.Yv-SynthActions.nTanh(vel.y/vxymax,in_p.v_nthfactor)).^2)).*attention,1),size(attention,2),numel(in_g.Xv))';
            hand_v = exp(-1/(2*in_p.v_var)*((in_g.Xv_hand-SynthActions.nTanh(vel.x_hand/vxymax,obj.Data.in_p.v_nthfactor)).^2 + (in_g.Yv_hand-SynthActions.nTanh(vel.y_hand/vxymax,in_p.v_nthfactor)).^2));
            att_object_hand_pos = reshape(sum(exp(-1/(2*in_p.pos_var)*((in_g.Xpos-SynthActions.nTanh((obj.getXFromSamples(samples)-obj.getXHandFromSamples(samples))/obj.Data.dims.scen_w,in_p.pos_nthfactor)).^2 + (in_g.Ypos-SynthActions.nTanh((obj.getYFromSamples(samples)-obj.getYHandFromSamples(samples))/obj.Data.dims.scen_h,in_p.pos_nthfactor)).^2)).*attention,1),size(attention,2),numel(in_g.Xpos))';
            input = [att_object_id;open_close;att_object_v;hand_v;att_object_hand_pos];
        end
        % Update state variables required for scenario visualization
        function updateVisStt(obj)
            obj.VisStt.last.traj.x = obj.VisStt.traj.x(:,end-size(obj.Stt.nxtsmpls,2));
            obj.VisStt.last.traj.y = obj.VisStt.traj.y(:,end-size(obj.Stt.nxtsmpls,2));
            obj.VisStt.last.traj.x_hand = obj.VisStt.traj.x_hand(:,end-size(obj.Stt.nxtsmpls,2));
            obj.VisStt.last.traj.y_hand = obj.VisStt.traj.y_hand(:,end-size(obj.Stt.nxtsmpls,2));
            obj.VisStt.last.traj.hand_h = obj.VisStt.traj.hand_h(:,end-size(obj.Stt.nxtsmpls,2));
            obj.VisStt.traj.x = obj.VisStt.traj.x(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
            obj.VisStt.traj.y = obj.VisStt.traj.y(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
            obj.VisStt.traj.x_hand = obj.VisStt.traj.x_hand(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
            obj.VisStt.traj.y_hand = obj.VisStt.traj.y_hand(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
            obj.VisStt.traj.hand_h = obj.VisStt.traj.hand_h(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
            if obj.Features
                obj.VisStt.last.att_G.A = obj.VisStt.att_G.A(:,end-size(obj.Stt.nxtsmpls,2));
                obj.VisStt.last.att_G.a = obj.VisStt.att_G.a(:,end-size(obj.Stt.nxtsmpls,2));
                obj.VisStt.last.att_G.b = obj.VisStt.att_G.b(:,end-size(obj.Stt.nxtsmpls,2));
                obj.VisStt.last.att_G.c = obj.VisStt.att_G.c(:,end-size(obj.Stt.nxtsmpls,2));
                obj.VisStt.last.att_G.x0 = obj.VisStt.att_G.x0(:,end-size(obj.Stt.nxtsmpls,2));
                obj.VisStt.last.att_G.y0 = obj.VisStt.att_G.y0(:,end-size(obj.Stt.nxtsmpls,2));
                obj.VisStt.att_G.A = obj.VisStt.att_G.A(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
                obj.VisStt.att_G.a = obj.VisStt.att_G.a(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
                obj.VisStt.att_G.b = obj.VisStt.att_G.b(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
                obj.VisStt.att_G.c = obj.VisStt.att_G.c(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
                obj.VisStt.att_G.x0 = obj.VisStt.att_G.x0(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
                obj.VisStt.att_G.y0 = obj.VisStt.att_G.y0(:,end-size(obj.Stt.nxtsmpls,2)+1:end);
            end
        end
    end
    
    % Sample generation abstract methods
    methods (Abstract,Access = protected)
        % Current scenario state
        [objtypes,traj,lbls] = getTrajectories(obj)
    end
    
    % Sample generation static methods
    methods (Static,Access = protected)
        % Add white noise to the input
        function input = addNoise(input,sigma_n)
            input = input + sigma_n*randn(size(input));
            input(input > 1) = 1;
            input(input < 0) = 0;
        end
        % Low-pass filter the samples
        function samples = smoothSamples(samples)
            for i = 1:size(samples,1)
                samples(i,:) = smooth(samples(i,:),5)';
            end
        end
        % Non-linear mapping in [-1,1] that compresses large values and
        % expands small values
        function nthval = nTanh(val,nthfactor)
            nthval = tanh(nthfactor*val)/tanh(nthfactor);
        end
    end
    
    % Visualization methods
    methods
        % Scenario Plot Generation
        function scenario_plot = getScenarioPlot(obj)
            x = round(obj.VisStt.last.traj.x+obj.VisData.margin-obj.Data.dims.obj_w/2);
            y = round(obj.VisStt.last.traj.y+obj.VisData.margin-obj.Data.dims.obj_h/2);
            %objs_dims = ones(length(objects_types),2).*[dims.obj_w,dims.obj_h];
            %scenario = insertShape(zeros(dims.scen_h+2*margin,dims.scen_w+2*margin,3),'FilledRectangle',[x' y' objs_dims],'Color',color_strs(objects_types)); % very slow
            scenario_plot = zeros(obj.Data.dims.scen_h+2*obj.VisData.margin,obj.Data.dims.scen_w+2*obj.VisData.margin,3);
            for i = 1:obj.Data.n_objects
                scenario_plot(y(i):y(i)+round(obj.Data.dims.obj_h)-1,x(i):x(i)+obj.Data.dims.obj_w-1,:) = obj.VisData.color_objs(:,:,:,obj.VisStt.objtypes(i));
            end
            x = round(obj.VisStt.last.traj.x_hand+obj.VisData.margin-obj.VisData.hand_w/2);
            y = round(obj.VisStt.last.traj.y_hand+obj.VisData.margin-obj.VisStt.last.traj.hand_h/2);
            scenario_plot(y:y+round(obj.VisStt.last.traj.hand_h)-1,x:x+obj.VisData.hand_w-1,:) = 1;
            if obj.Features
                att_A = obj.VisStt.last.att_G.A;
                att_a = obj.VisStt.last.att_G.a;
                att_b = obj.VisStt.last.att_G.b;
                att_c = obj.VisStt.last.att_G.c;
                att_x0 = obj.VisStt.last.att_G.x0;
                att_y0 = obj.VisStt.last.att_G.y0;
                att_vis = att_A*exp(-(att_a*(obj.VisData.Xscenario-att_x0-obj.VisData.margin).^2 + 2*att_b*(obj.VisData.Xscenario-att_x0-obj.VisData.margin).*(obj.VisData.Yscenario-att_y0-obj.VisData.margin) + att_c*(obj.VisData.Yscenario-att_y0-obj.VisData.margin).^2));
                scenario_plot = scenario_plot + att_vis/obj.Data.att_p.A_factor;
                scenario_plot(scenario_plot > 1) = 1;
            end
        end
        % Raw input plot generation
        function input_plot = getRawInputPlot(obj,input)
            x = round(obj.getXFromSamples(input)+obj.VisData.margin-obj.Data.dims.obj_w/2);
            y = round(obj.getYFromSamples(input)+obj.VisData.margin-obj.Data.dims.obj_h/2);
            input_plot = zeros(obj.Data.dims.scen_h+2*obj.VisData.margin,obj.Data.dims.scen_w+2*obj.VisData.margin,3);
            object_ids = obj.getObjectIdsFromSamples(input);
            for i = 1:obj.Data.n_objects
                object_id = object_ids((i-1)*obj.Data.in_p.objid_sz+1:i*obj.Data.in_p.objid_sz);
                divs = [0,round(obj.Data.in_p.objid_sz/3),round(2*obj.Data.in_p.objid_sz/3),obj.Data.in_p.objid_sz];
                color_obj = zeros(1,1,3);
                for j = 1:length(color_obj)
                    color_obj(j) = sum(object_id(divs(j)+1:divs(j+1)));
                end
                input_plot(y(i):y(i)+round(obj.Data.dims.obj_h)-1,x(i):x(i)+obj.Data.dims.obj_w-1,:) = color_obj.*ones(obj.Data.dims.obj_h,obj.Data.dims.obj_w);
            end
            x = round(obj.getXHandFromSamples(input)+obj.VisData.margin-obj.VisData.hand_w/2);
            y = round(obj.getYHandFromSamples(input)+obj.VisData.margin-obj.getHandHFromSamples(input)/2);
            input_plot(y:y+round(obj.getHandHFromSamples(input))-1,x:x+obj.VisData.hand_w-1,:) = 1;
            input_plot(input_plot > 1) = 1;
        end
        % Input plot generation
        function input_plot = getInputPlot(obj,input)
            if obj.Features
                input_plot = SynthActions.getInputPlotStatic(input,obj.Data.in_p);
            else
                input_plot = obj.getRawInputPlot(input);
            end
        end
    end
    
    % Static visualization methods
    methods (Static)
        % Input plot generation (color code: attention object id in red,
        % open/close-ness of hand in green, attention object velocity in
        % yellow, hand velocity in cyan and attention object position wrt
        % hand in blue)
        function input_plot = getInputPlotStatic(input,in_p)
            input_plot = zeros(2+max(2+in_p.pos_sd,2*in_p.v_sd+1),max(in_p.objid_sz,in_p.pos_sd+in_p.v_sd+1),3);
            vis_pss = [1,1;3,1;size(input_plot,1)-2*in_p.v_sd,size(input_plot,2)-in_p.v_sd+1;size(input_plot,1)-in_p.v_sd+1,size(input_plot,2)-in_p.v_sd+1;size(input_plot,1)-in_p.pos_sd+1,1];
            vis_szs = [1,in_p.objid_sz;1,in_p.opcl_sz;in_p.v_sd,in_p.v_sd;in_p.v_sd,in_p.v_sd;in_p.pos_sd,in_p.pos_sd];
            vis_nds = vis_pss + vis_szs - 1;
            inlimits = [0;cumsum(prod(vis_szs,2))];
            colors = permute([1,0,0;0,1,0;1,1,0;0,1,1;0,0,1],[1,3,2]);
            for i = 1:size(vis_pss,1)
                input_plot(vis_pss(i,1):vis_nds(i,1),vis_pss(i,2):vis_nds(i,2),:) = colors(i,:,:).*reshape(input(inlimits(i)+1:inlimits(i+1)),vis_szs(i,1),vis_szs(i,2));
            end
        end
    end
    
end