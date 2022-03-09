function [Ac_SKLDA,code_dvtest_sklda,code_latest_sklda]=SKLDA_xx0(Train_trials,Train_label,Test_trials,Test_label)
%% SKLDA 
fprintf('Now Algorithm is SKLDA......\n');
MdlLinear = fitcdiscr(Train_trials,Train_label,'SaveMemory','on','FillCoeffs','off');
rng('default') 
[err,gamma,delta,numpred] = cvshrink(MdlLinear,'NumGamma',24,'Verbose',1);% NumGamma--µü´ú´ÎÊý
[r,s]=find(err==min(err));
MdlLinear.Gamma = gamma(r(1));%gamma;
[meanclass pv] = predict(MdlLinear,Test_trials);
Ac_SKLDA=sum(meanclass==Test_label)./size(Test_trials,1);
code_dvtest_sklda=pv(:,2)-pv(:,1);
code_latest_sklda=[zeros(size(Test_trials,1)/2,1)-1;ones(size(Test_trials,1)/2,1)];