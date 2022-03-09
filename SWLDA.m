function [Ac_SWLDA,code_dvtest_swlda,code_latest_swlda]=SWLDA(Train_trials,Train_label,Test_trials,Test_label,SW_Num)
%% SWLDA
fprintf('Now Algorithm is SWLDA......\n');
[b,se,pval,inmodel,stats,nextstep,history]=stepwisefit(Train_trials,Train_label,'penter',0.1,'premove',0.15,'display','off','maxiter',SW_Num);
if sum(inmodel)==0
    inmodel=~inmodel;
end
MdlLinear = fitcdiscr(Train_trials(:,inmodel),Train_label);
[meanclass pv] = predict(MdlLinear,Test_trials(:,inmodel));
Ac_SWLDA=sum(meanclass==Test_label)./size(Test_trials,1);
code_dvtest_swlda=pv(:,2)-pv(:,1);
code_latest_swlda=[zeros(size(Test_trials,1)/2,1)-1;ones(size(Test_trials,1)/2,1)];