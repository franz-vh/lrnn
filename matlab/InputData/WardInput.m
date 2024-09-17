% WARD (Wearable Action Recognition Database) dataset input class

classdef WardInput < MultTrialDataSet
    
    % Public constants
    properties (Constant)
        Path = strcat(fileparts(mfilename('fullpath')),"/../../datasets/WARD1.0")
        Classifs = createEnum({'actions','subjects'}) % classifications enumeration
        ActionLbls = createEnum({'rest','resi','reli','wafo','wale','wari','tule','turi','up','down','jog','jump','push'}) % action labels enumeration
        SubjectLbls = createEnum({'one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty'}) % subject labels enumeration
        AllLbls = {WardInput.ActionLbls,WardInput.SubjectLbls} % cell with all labels enumerations
    end
    
    % Public constants
    properties (SetAccess = protected)
        AllNLbls = InputData.setAllNLbls(WardInput.AllLbls)
    end
    
    % Data initialization methods
    methods (Access = protected)
        % Read dataset
        function readDataset(obj)
            subdirnames = dir(fullfile(WardInput.Path,"Subject*"));
            subtrials = cell(1,length(subdirnames));
            ntrials = 0;
            for i = 1:length(subdirnames)
                filenameparts = split(subdirnames(i).name,'Subject');
                lbl_subject = str2double(filenameparts{2});
                filenames = dir(fullfile(WardInput.Path,subdirnames(i).name,"*.mat"));
                subtrials{i} = cell(1,length(filenames));
                ntrials = ntrials + length(filenames);
                for j = 1:length(filenames)
                    trial = open(fullfile(filenames(j).folder,filenames(j).name)).WearableData;
                    subtrials{i}{j}.x = cell2mat(trial.Reading)';
                    lbl_action = trial.Class;
                    if ischar(lbl_action) % sometimes the label is as a number, sometimes as a string
                        lbl_action = str2double(lbl_action);
                    end
                    subtrials{i}{j}.lbl = zeros(length(WardInput.AllLbls),1);
                    subtrials{i}{j}.lbl(WardInput.Classifs.actions) = lbl_action;
                    subtrials{i}{j}.lbl(WardInput.Classifs.subjects) = lbl_subject;
                end
            end
            obj.Data.trials = cell(1,ntrials);
            k = 1;
            for i = 1:length(subtrials)
                for j = 1:length(subtrials{i})
                    obj.Data.trials{k} = subtrials{i}{j};
                    k = k + 1;
                end
            end
        end
    end
    
end