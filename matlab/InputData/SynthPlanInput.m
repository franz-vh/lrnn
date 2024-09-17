% Synthetically generated plans input class

classdef SynthPlanInput < SynthPlans & CompoundInput
    
    % Public constants
    properties (Constant)
        Features = true % handcrafted features data
        SubInputs = createEnum({'hand','environment'}) % subinputs enumeration
    end
    
    % Main public methods
    methods
        % Constructor
        function obj = SynthPlanInput(~)
            obj@CompoundInput();
            obj@SynthPlans();
        end
    end
    
    % Data initialization methods
    methods (Access = protected)
        % Input data initialization
        function initData(obj)
            obj.initData@SynthActions();
            obj.SubDims{SynthPlanInput.SubInputs.hand} = obj.Size;
            data = obj.Data;
            % Environment subinput parameters
            data.in_p.env_h = 10;
            data.in_p.env_w = 10;
            obj.SubDims{SynthPlanInput.SubInputs.environment} = [data.in_p.env_h,data.in_p.env_w];
            obj_sigma = min(data.dims.scen_h,data.dims.scen_w)/12;
            data.in_p.obj_var = obj_sigma^2;
            % Environment subinput generation constants
            [Xenv,Yenv] = meshgrid(linspace(1,data.dims.scen_h,data.in_p.env_h),linspace(1,data.dims.scen_w,data.in_p.env_w));
            data.in_g.Xenv = reshape(Xenv,1,1,numel(Xenv));
            data.in_g.Yenv = reshape(Yenv,1,1,numel(Yenv));
            % Input parameters
            obj.Data = data;
            obj.initData@CompoundInput();
        end
    end
    
    % Sample generation methods
    methods (Access = protected)
        % Generate input data
        function input = generateInput(obj,samples,vel,attention)
            in_g = obj.Data.in_g;
            in_p = obj.Data.in_p;
            env = reshape(sum(exp(-1/(2*in_p.obj_var)*((in_g.Xenv-obj.getXFromSamples(samples)).^2+(in_g.Yenv-obj.getYFromSamples(samples)).^2)),1),size(samples,2),numel(in_g.Xenv))';
            env(env > 1) = 1;
            input = [obj.generateInput@SynthActions(samples,vel,attention);env];
        end
    end
    
    % Visualization methods
    methods
        % Input plot generation
        function input_plot = getInputPlot(obj,input)
            input_plot = obj.getInputPlot@CompoundInput(input);
        end
        % Subinput plot generation
        function input_plot = getSubInputPlot(obj,input,subinput)
            if subinput == SynthPlanInput.SubInputs.hand
                input_plot = SynthActions.getInputPlotStatic(input,obj.Data.in_p);
            else
                input_plot = obj.getSubInputPlot@CompoundInput(input,subinput);
            end
        end
    end
    
end