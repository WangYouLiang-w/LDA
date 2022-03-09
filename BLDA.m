function [Ac_BLDA,code_dvtest_blda,code_latest_blda]=BLDA_xx0(Train_trials,Train_label,Test_trials,Test_label,n_channels)
%% BLDA
fprintf('Now Algorithm is BLDA......\n');
setpath();
% n_channels = 10;
tx = reshape(Train_trials,size(Train_trials,1),n_channels,[]);
tx = permute(tx,[2,3,1]);
w = windsor;
w = train(w,tx,0.1);
tx = apply(w,tx);
n = normalize;
n = train(n,tx,'z-score');
tx = apply(n,tx);    
n_samples = size(tx,2);
n_trials = size(tx,3);
tx = reshape(tx,n_samples*n_channels,n_trials);
ty=Train_label';
bayes = bayeslda(1);
bayes = train(bayes,tx,ty);  
for ni = 1:size(Test_trials,1)
    tx = reshape(squeeze(Test_trials(ni,:))',n_channels,[]);
    tx = apply(w,tx);
    tx = apply(n,tx);
    tx = reshape(tx,[],1);
    ty = classify(bayes,tx);
    code_dvtest_blda(ni)=ty;
    if ty>0
        meanclass(ni) = 1;
    else
        meanclass(ni) = -1;
    end
end
count=sum(meanclass'==Test_label);
Ac_BLDA=sum(count)./size(Test_trials,1);
code_latest_blda=[zeros(size(Test_trials,1)/2,1)-1;ones(size(Test_trials,1)/2,1)];
